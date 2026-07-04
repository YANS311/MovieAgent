# 文件: myapp/utils/pagination.py (V7 - 终极合并版)

from django.utils.safestring import mark_safe
from django.http.request import QueryDict
import copy


class Pagination(object):
    # 1. 这里的 __init__ 必须包含 exclude_params=None
    def __init__(self, request, queryset, page_size=10, page_param="page", plus=5, exclude_params=None):

        # 深度拷贝 GET 参数
        query_dict = copy.deepcopy(request.GET)
        query_dict._mutable = True

        # --- 修复 1: 剔除互斥参数 (用于多 Tab 分页不冲突) ---
        if exclude_params:
            for param in exclude_params:
                if param in query_dict:
                    del query_dict[param]
        # ------------------------------------------------

        self.query_dict = query_dict
        self.page_param = page_param

        page = request.GET.get(page_param, "1")
        if page.isdecimal():
            page = int(page)
        else:
            page = 1

        self.page = page
        self.page_size = page_size
        self.start = (page - 1) * page_size
        self.end = (page) * page_size

        # 切片
        self.page_queryset = queryset[self.start:self.end]

        # --- 修复 2: 正确区分 List 和 QuerySet 的计数方式 ---
        if isinstance(queryset, list):
            # 如果是 Python 列表，用 len()
            self.total_count = len(queryset)
        else:
            # 否则假设是 Django QuerySet，用 count()
            self.total_count = queryset.count()
        # --------------------------------------------------

        total_page_count, div = divmod(self.total_count, page_size)
        if div:
            total_page_count += 1
        self.total_page_count = total_page_count
        self.plus = plus

    def html(self):
        if self.total_page_count <= 2 * self.plus + 1:
            start_page = 1
            end_page = self.total_page_count
        else:
            if self.page <= self.plus:
                start_page = 1
                end_page = 2 * self.plus + 1
            else:
                if self.page + self.plus > self.total_page_count:
                    start_page = self.total_page_count - 2 * self.plus
                    end_page = self.total_page_count
                else:
                    start_page = self.page - self.plus
                    end_page = self.page + self.plus

        page_str_list = []

        # 首页
        self.query_dict.setlist(self.page_param, [1])
        page_str_list.append(f'<li><a href="?{self.query_dict.urlencode()}">首页</a></li>')

        # 上一页
        if self.page > 1:
            self.query_dict.setlist(self.page_param, [self.page - 1])
            prev = f'<li><a href="?{self.query_dict.urlencode()}">上一页</a></li>'
        else:
            self.query_dict.setlist(self.page_param, [1])
            prev = f'<li class="disabled"><a href="?{self.query_dict.urlencode()}">上一页</a></li>'
        page_str_list.append(prev)

        # 页码
        for i in range(start_page, end_page + 1):
            self.query_dict.setlist(self.page_param, [i])
            if i == self.page:
                prev = f'<li class="active"><a href="?{self.query_dict.urlencode()}">{i}</a></li>'
            else:
                prev = f'<li><a href="?{self.query_dict.urlencode()}">{i}</a></li>'
            page_str_list.append(prev)

        # 下一页
        if self.page < self.total_page_count:
            self.query_dict.setlist(self.page_param, [self.page + 1])
            prev = f'<li><a href="?{self.query_dict.urlencode()}">下一页</a></li>'
        else:
            self.query_dict.setlist(self.page_param, [self.total_page_count])
            prev = f'<li class="disabled"><a href="?{self.query_dict.urlencode()}">下一页</a></li>'
        page_str_list.append(prev)

        # 尾页
        self.query_dict.setlist(self.page_param, [self.total_page_count])
        page_str_list.append(f'<li><a href="?{self.query_dict.urlencode()}">尾页</a></li>')

        # 跳转框
        hidden_inputs = ""
        for key, value_list in self.query_dict.lists():
            if key == self.page_param:
                continue
            for value in value_list:
                if value:
                    hidden_inputs += f'<input type="hidden" name="{key}" value="{value}">'

        search_string = f"""
        <li>
            <form style="float: left; margin-left: -1px" method="GET">
                {hidden_inputs} 
                <div class="input-group" style="width: 200px">
                    <input type="text" name="{self.page_param}" 
                           style="position: relative;float: left;display: inline-block; width: 88px;border-radius: 0;" 
                           class="form-control" placeholder="页码">
                    <span class="input-group-btn">
                        <button style="border-radius: 0" class="btn btn-default" type="submit">Go</button>
                    </span>
                </div>
            </form>
        </li>
        """
        page_str_list.append(search_string)
        page_string = mark_safe("".join(page_str_list))
        return page_string