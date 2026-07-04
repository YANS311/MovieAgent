import pandas as pd
import numpy as np
import ast
import gc


def parse_crew(x):
    # 提取导演
    try:
        if pd.isna(x): return ""
        x = ast.literal_eval(x)
        for item in x:
            if item['job'] == 'Director':
                return item['name']
        return ""
    except:
        return ""


def parse_cast(x):
    # 提取前3名演员
    try:
        if pd.isna(x): return ""
        x = ast.literal_eval(x)
        # 只取前3个，用竖线分隔
        names = [item['name'] for item in x[:3]]
        return "|".join(names)
    except:
        return ""


def main():
    print("--- 🚀 Step 1: Data Merging & Cleaning ---")

    # 1. 加载 MovieLens 链接文件 (这是桥梁)
    print("1. Loading ML-32M Links...")
    links = pd.read_csv('./ml-32m/links.csv', dtype={'tmdbId': 'str'})
    links = links.dropna(subset=['tmdbId'])  # 没TMDB ID的没法匹配，丢弃
    # 去除小数点 (e.g., 123.0 -> 123)
    links['tmdbId'] = links['tmdbId'].apply(lambda x: x.split('.')[0])
    print(f"   -> Valid Links: {len(links)}")

    # 2. 加载 Kaggle Metadata (简介)
    print("2. Loading Movies Metadata (Overview)...")
    # low_memory=False 防止混合类型警告
    meta = pd.read_csv('./the-movies-dataset/movies_metadata.csv', low_memory=False)

    # 清洗 meta 的 ID
    meta = meta[['id', 'overview', 'title']].copy()
    # 有些脏数据的 id 是日期，强制转错为 NaN
    meta['id'] = pd.to_numeric(meta['id'], errors='coerce')
    meta = meta.dropna(subset=['id'])
    meta['id'] = meta['id'].astype(int).astype(str)  # 转成字符串方便合并

    # 3. 加载 Kaggle Credits (演职员)
    print("3. Loading Credits (Cast & Director)...")
    credits = pd.read_csv('./the-movies-dataset/credits.csv')
    credits['id'] = credits['id'].astype(str)

    # 提取导演和演员 (这一步比较慢，大概几分钟)
    print("   -> Parsing Cast/Crew JSON (Please wait)...")
    credits['director'] = credits['crew'].apply(parse_crew)
    credits['cast'] = credits['cast'].apply(parse_cast)

    # 只保留处理后的列
    credits = credits[['id', 'director', 'cast']]

    # 4. 级联像 (Merge)
    print("4. Merging Tables...")

    # 先把 Meta 和 Credits 合并 (Based on TMDB ID)
    kaggle_data = pd.merge(meta, credits, on='id', how='left')

    # 再把 MovieLens 和 Kaggle 数据合并
    # links['tmdbId'] <--> kaggle['id']
    final_df = pd.merge(links, kaggle_data, left_on='tmdbId', right_on='id', how='left')

    # 5. 后处理
    # 填充缺失值
    final_df['overview'] = final_df['overview'].fillna('')
    final_df['director'] = final_df['director'].fillna('Unknown')
    final_df['cast'] = final_df['cast'].fillna('')

    # 只要 movieId 和新特征
    final_df = final_df[['movieId', 'title', 'overview', 'director', 'cast']]

    # 保存
    output_file = 'ml32m_enhanced_meta.csv'
    final_df.to_csv(output_file, index=False)
    print(f"✅ Data Merged! Saved to {output_file}")
    print(final_df.head())


if __name__ == "__main__":
    main()