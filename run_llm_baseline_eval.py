#!/usr/bin/env python3
"""
LLM 基座模型满血对比实验
"""
import json, time, requests, csv, math
from collections import defaultdict

MODELS = ["qwen3:0.6b","qwen3:4b-instruct","qwen3-vl:4b","qwen3:8b","qwen3.5:9b"]
OLLAMA_URL = "http://localhost:11434/api/chat"
SYSTEM_PROMPT = """你是一个专业的电影推荐助手。根据用户的查询，推荐5部最合适的电影。
请严格按照JSON格式输出：{"recommendations":["电影名1","电影名2","电影名3","电影名4","电影名5"]}
只输出JSON，不要其他内容。"""

def load_data():
    with open("myapp/management/commands/golden_dataset_agent_eval.json","r",encoding="utf-8") as f:
        return json.load(f)["queries"]

def call_ollama(model,query,timeout=180):
    p={"model":model,"messages":[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":query}],"stream":False,"options":{"temperature":0,"num_predict":512}}
    t0=time.time()
    try:
        r=requests.post(OLLAMA_URL,json=p,timeout=timeout); r.raise_for_status()
        return r.json().get("message",{}).get("content",""), time.time()-t0
    except Exception as e:
        return f"ERR:{e}", time.time()-t0

def parse(text):
    try:
        s=text.find("{"); e=text.rfind("}")+1
        if s>=0 and e>s: return json.loads(text[s:e]).get("recommendations",[])
    except: pass
    return []

def hr(rec,gt):
    for r in rec:
        for g in gt:
            if g in r or r in g: return 1.0
    return 0.0

def ndcg(rec,gt,k=5):
    dcg=0.0
    for i,r in enumerate(rec[:k]):
        for g in gt:
            if g in r or r in g: dcg+=1.0/math.log2(i+2); break
    n=min(len(gt),k); idcg=sum(1.0/math.log2(i+2) for i in range(n))
    return dcg/idcg if idcg>0 else 0.0

def main():
    queries=load_data()
    print(f"总查询数: {len(queries)}")
    results=[]
    for model in MODELS:
        print(f"\n{'='*50}\n  Model: {model}\n{'='*50}")
        hrs,ndcgs,lats=[],[],[]
        per_d=defaultdict(lambda:{"hr":[],"ndcg":[]})
        for i,q in enumerate(queries):
            gt=q["ground_truth_movies"]; d=q["difficulty"]
            txt,lat=call_ollama(model,q["query"])
            rec=parse(txt)
            h=hr(rec,gt); n=ndcg(rec,gt)
            hrs.append(h); ndcgs.append(n); lats.append(lat)
            per_d[d]["hr"].append(h); per_d[d]["ndcg"].append(n)
            if (i+1)%10==0 or i==0: print(f"  [{i+1}/{len(queries)}] HR={h:.1f} lat={lat:.1f}s")
        ah=sum(hrs)/len(hrs); an=sum(ndcgs)/len(ndcgs); al=sum(lats)/len(lats)
        row={"model":model,"HR@5":f"{ah:.4f}","NDCG@5":f"{an:.4f}","Latency":f"{al:.2f}"}
        for d in per_d:
            dd=per_d[d]
            row[f"HR_{d}"]=f"{sum(dd['hr'])/len(dd['hr']):.4f}" if dd["hr"] else "0"
            row[f"NDCG_{d}"]=f"{sum(dd['ndcg'])/len(dd['ndcg']):.4f}" if dd["ndcg"] else "0"
        results.append(row)
        print(f"  => HR@5={ah:.4f} NDCG@5={an:.4f} Lat={al:.2f}s")
        for d in per_d:
            dd=per_d[d]
            print(f"    {d}: HR={sum(dd['hr'])/len(dd['hr']):.4f}")
    out="experiment_results/llm_baseline_full.csv"
    with open(out,"w",newline="",encoding="utf-8-sig") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0].keys()))
        w.writeheader(); w.writerows(results)
    print(f"\n  => Saved {out}")

if __name__=="__main__":
    main()