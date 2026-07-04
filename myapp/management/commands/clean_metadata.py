import django
from django.core.management.base import BaseCommand
from django.db import transaction
from myapp.models import Movie, Genre, Region


class Command(BaseCommand):
    help = '清洗 Genre 和 Region 数据，合并中英文混杂的标签'

    def handle(self, *args, **options):
        self.stdout.write("--- 🧹 开始数据清洗 (V2: 完整映射版) ---")

        # =========================================
        # 1. 类型映射 (Genre Mapping)
        # =========================================
        genre_map = {
            'Action': '动作',
            'Adventure': '冒险',
            'Animation': '动画',
            'Children\'s': '儿童',  # 数据库中没有“儿童”时，会自动将“Children\'s”改名为“儿童”
            'Comedy': '喜剧',
            'Crime': '犯罪',
            'Documentary': '纪录',
            'Drama': '剧情',
            'Fantasy': '奇幻',
            'Film-Noir': '黑色电影',
            'Horror': '恐怖',
            'Musical': '音乐',
            'Mystery': '悬疑',
            'Romance': '爱情',
            'Sci-Fi': '科幻',
            'Thriller': '惊悚',
            'War': '战争',
            'Western': '西部'
        }

        # =========================================
        # 2. 地区映射 (Region Mapping)
        # =========================================
        region_map = {
            # 常见大国
            'USA': '美国', 'United States': '美国', 'America': '美国',
            'UK': '英国', 'United Kingdom': '英国', 'Great Britain': '英国',
            'France': '法国',
            'Japan': '日本',
            'Korea': '韩国', 'South Korea': '韩国',
            'Germany': '德国', 'West Germany': '德国', 'East Germany': '东德',
            'Italy': '意大利',
            'Spain': '西班牙',
            'India': '印度',
            'Canada': '加拿大',
            'Australia': '澳大利亚',
            'Russia': '俄罗斯',
            'Soviet Union': '苏联',
            'Brazil': '巴西',
            'Mexico': '墨西哥',

            # 亚洲/大洋洲
            'China': '中国大陆', 'Hong Kong': '中国香港', 'Taiwan': '中国台湾',
            'Thailand': '泰国',
            'Vietnam': '越南',
            'Mongolia': '蒙古',
            'Philippines': '菲律宾',
            'Singapore': '新加坡',
            'Malaysia': '马来西亚',
            'Indonesia': '印度尼西亚',
            'Iran': '伊朗',
            'Israel': '以色列',
            'Turkey': '土耳其',
            'New Zealand': '新西兰',
            'Pakistan': '巴基斯坦',
            'Afghanistan': '阿富汗',
            'Kazakhstan': '哈萨克斯坦',
            'Uzbekistan': '乌兹别克斯坦',
            'Tajikistan': '塔吉克斯坦',
            'Nepal': '尼泊尔',
            'Bhutan': '不丹',
            'Cambodia': '柬埔寨',
            'Lebanon': '黎巴嫩',
            'Jordan': '约旦',
            'Kuwait': '科威特',
            'Qatar': '卡塔尔',
            'Bahrain': '巴林',
            'Saudi Arabia': '沙特阿拉伯',
            'United Arab Emirates': '阿联酋',
            'Iraq': '伊拉克',
            'Georgia': '格鲁吉亚',
            'Palestinian Territory': '巴勒斯坦',

            # 欧洲
            'Sweden': '瑞典',
            'Switzerland': '瑞士',
            'Netherlands': '荷兰',
            'Belgium': '比利时',
            'Austria': '奥地利',
            'Denmark': '丹麦',
            'Finland': '芬兰',
            'Norway': '挪威',
            'Ireland': '爱尔兰',
            'Poland': '波兰',
            'Czech Republic': '捷克', 'Czechoslovakia': '捷克斯洛伐克',
            'Hungary': '匈牙利',
            'Greece': '希腊',
            'Portugal': '葡萄牙',
            'Ukraine': '乌克兰',
            'Romania': '罗马尼亚',
            'Bulgaria': '保加利亚',
            'Iceland': '冰岛',
            'Serbia': '塞尔维亚', 'Serbia and Montenegro': '塞尔维亚和黑山',
            'Croatia': '克罗地亚',
            'Bosnia and Herzegovina': '波黑',
            'Slovakia': '斯洛伐克',
            'Slovenia': '斯洛文尼亚',
            'Lithuania': '立陶宛',
            'Latvia': '拉脱维亚',
            'Estonia': '爱沙尼亚',
            'Luxembourg': '卢森堡',
            'Monaco': '摩纳哥',
            'Liechtenstein': '列支敦士登',
            'Malta': '马耳他',
            'Cyprus': '塞浦路斯',
            'Albania': '阿尔巴尼亚',
            'Macedonia': '马其顿',
            'Montenegro': '黑山',
            'Yugoslavia': '南斯拉夫',

            # 美洲
            'Argentina': '阿根廷',
            'Chile': '智利',
            'Colombia': '哥伦比亚',
            'Peru': '秘鲁',
            'Venezuela': '委内瑞拉',
            'Cuba': '古巴',
            'Uruguay': '乌拉圭',
            'Bolivia': '玻利维亚',
            'Ecuador': '厄瓜多尔',
            'Jamaica': '牙买加',
            'Bahamas': '巴哈马',
            'Puerto Rico': '波多黎各',
            'Haiti': '海地',
            'Aruba': '阿鲁巴',
            'Nicaragua': '尼加拉瓜',
            'Netherlands Antilles': '荷属安的列斯',
            'St. Kitts and Nevis': '圣基茨和尼维斯',

            # 非洲
            'South Africa': '南非',
            'Egypt': '埃及',
            'Morocco': '摩洛哥',
            'Algeria': '阿尔及利亚',
            'Tunisia': '突尼斯',
            'Kenya': '肯尼亚',
            'Nigeria': '尼日利亚',
            'Ethiopia': '埃塞俄比亚',
            'Ghana': '加纳',
            'Cameroon': '喀麦隆',
            'Senegal': '塞内加尔',
            'Namibia': '纳米比亚',
            'Botswana': '博茨瓦纳',
            'Rwanda': '卢旺达',
            'Tanzania': '坦桑尼亚',
            'Congo': '刚果',
            'Cote D\'Ivoire': '科特迪瓦',
            'Burkina Faso': '布基纳法索',
            'Mauritania': '毛里塔尼亚',
            'Libyan Arab Jamahiriya': '利比亚',
        }

        # =========================================
        # 3. 执行清洗
        # =========================================

        # 预处理：去除首尾空格
        self.clean_whitespace(Region, "地区")
        self.clean_whitespace(Genre, "类型")

        # 执行合并
        self.merge_tags(Region, region_map, "地区")
        self.merge_tags(Genre, genre_map, "类型")

        self.stdout.write(self.style.SUCCESS("√ 所有数据清洗完成！"))

    def clean_whitespace(self, ModelClass, tag_name):
        self.stdout.write(f"正在去除 {tag_name} 空格...")
        count = 0
        for obj in ModelClass.objects.all():
            stripped = obj.name.strip()
            if obj.name != stripped:
                # 检查改名后是否冲突
                if ModelClass.objects.filter(name=stripped).exists():
                    # 冲突了，先把关联转移过去，再删除自己
                    target = ModelClass.objects.get(name=stripped)
                    if tag_name == "地区":
                        for m in obj.movie_set.all():
                            m.regions.add(target)
                            m.regions.remove(obj)
                    else:
                        for m in obj.movie_set.all():
                            m.genres.add(target)
                            m.genres.remove(obj)
                    obj.delete()
                else:
                    obj.name = stripped
                    obj.save()
                count += 1
        self.stdout.write(f"   -> 清理了 {count} 个带空格的标签")

    def merge_tags(self, ModelClass, mapping, tag_name):
        self.stdout.write(f"\n正在合并 {tag_name} ...")
        count = 0

        # 开启事务，保证安全
        with transaction.atomic():
            for eng_name, cn_name in mapping.items():
                try:
                    # 1. 查找源标签 (英文) - 忽略大小写
                    source_obj = ModelClass.objects.filter(name__iexact=eng_name).first()
                    if not source_obj:
                        continue

                        # 2. 查找目标标签 (中文)
                    target_obj = ModelClass.objects.filter(name=cn_name).first()

                    if target_obj:
                        # 情况 A: 中英文都存在 -> 迁移关系后删除英文
                        if source_obj == target_obj: continue  # 自己映射自己

                        movies = source_obj.movie_set.all()
                        if movies.exists():
                            self.stdout.write(f"   🔄 合并: {source_obj.name} ({movies.count()}部) -> {target_obj.name}")
                            for m in movies:
                                if tag_name == "地区":
                                    m.regions.add(target_obj)
                                    m.regions.remove(source_obj)
                                else:
                                    m.genres.add(target_obj)
                                    m.genres.remove(source_obj)

                        source_obj.delete()
                        count += 1
                    else:
                        # 情况 B: 只有英文，没有中文 -> 直接改名
                        # 检查改名后是否会撞车 (比如已有 "United States" 和 "USA"，改名可能导致重复)
                        if ModelClass.objects.filter(name=cn_name).exists():
                            # 理论上上面 target_obj 没找到不应该进这里，但为了保险
                            continue

                        self.stdout.write(f"   ✏️ 改名: {source_obj.name} -> {cn_name}")
                        source_obj.name = cn_name
                        source_obj.save()
                        count += 1

                except Exception as e:
                    self.stdout.write(f"   ⚠️ 处理 {eng_name} 时出错: {e}")

        self.stdout.write(f"   -> 累计处理 {count} 个 {tag_name} 标签")