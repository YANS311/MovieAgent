#!/usr/bin/env python
"""Quick script to run Pure LLM and Naive RAG baselines"""
import os, sys, time, json, math, requests
sys.path.insert(0, os.path.dirname(__file__))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'DjangoProject3.settings')

import django
django.setup()

from django.contrib.auth import get_user_model
from myapp.models import Movie
from myapp.recommender.recall import multi_channel_recall, hot_recall
from myapp import views

User = get_user_model()

DATASET_PATH = 'myapp/management/commands/golden_dataset_agent_eval.json'
with open(DATASET_PATH, 'r', encoding='utf-8') as f:
    dataset = json.load(f)
queries = dataset['queries'][:50]  # 50 queries

user = User.objects.get(id=47040)
llm_model = "qwen3:4b-instruct"

def call_llm(prompt, timeout=60):
    r = requests.post("http://localhost:11434/api/chat", json={
        "model": llm_model,
        "messages": [
            {"role": "system", "content": "你是电影推荐助手。根据用户需求推荐5部电影，只输出电影标题，用逗号分隔。"},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 256},
    }, timeout=timeout)
    return r.json().get("message", {}).get("content", "")

def extract_movie_ids_from_text(text):
    ids = []
    for m in Movie.objects.all():
        if m.title in text:
            ids.append(m.id)
        if len(ids) >= 5:
            break
    return ids

def hit_rate_at_k(rec, gt, k=5):
    return 1.0 if (set(rec[:k]) & gt) else 0.0

def ndcg_at_k(rec, gt, k=5):
    dcg = sum(1.0 / math.log2(i + 2) for i, mid in enumerate(rec[:k]) if mid in gt)
    n_rel = min(len(gt), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0

results = {}
for mode in ['pure_llm', 'naive_rag']:
    hr_list, ndcg_list, sem_list, lat_list = [], [], [], []
    print(f"\n=== {mode} ===")
    for i, q in enumerate(queries):
        query_text = q['query']
        gt_titles = set(q['ground_truth_movies'])
        gt_ids = set()
        for title in gt_titles:
            for m in Movie.objects.filter(title__icontains=title):
                gt_ids.add(m.id)
        
        t0 = time.time()
        
        if mode == 'pure_llm':
            resp = call_llm(query_text)
            rec_ids = extract_movie_ids_from_text(resp)
        else:  # naive_rag
            rag_res = getattr(views, 'RAG_RESOURCES', {})
            try:
                if rag_res and rag_res.get('vectorstore'):
                    docs = rag_res['vectorstore'].similarity_search(query_text, k=5)
                    rec_ids = []
                    for d in docs:
                        title = d.metadata.get('title', '')
                        for m in Movie.objects.filter(title__icontains=title)[:1]:
                            rec_ids.append(m.id)
                    rec_ids = rec_ids[:5]
                else:
                    rec_ids = []
            except:
                rec_ids = []
        
        latency = (time.time() - t0) * 1000
        
        hr = hit_rate_at_k(rec_ids, gt_ids)
        ndcg = ndcg_at_k(rec_ids, gt_ids)
        
        hr_list.append(hr)
        ndcg_list.append(ndcg)
        lat_list.append(latency)
        
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(queries)}] hr={hr:.1f} lat={latency:.0f}ms")
    
    avg_hr = sum(hr_list) / len(hr_list)
    avg_ndcg = sum(ndcg_list) / len(ndcg_list)
    avg_lat = sum(lat_list) / len(lat_list)
    results[mode] = {'HR@5': avg_hr, 'NDCG@5': avg_ndcg, 'Latency': avg_lat}
    print(f"  {mode}: HR@5={avg_hr:.4f}, NDCG@5={avg_ndcg:.4f}, Latency={avg_lat:.0f}ms")

# Print summary
print("\n" + "=" * 60)
print("BASELINE SUMMARY")
print("=" * 60)
for mode, r in results.items():
    print(f"  {mode}: HR@5={r['HR@5']:.4f}, NDCG@5={r['NDCG@5']:.4f}, Latency={r['Latency']:.0f}ms")
