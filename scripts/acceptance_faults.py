"""Controlled faults on independent local service; no GPU allocation attack or DB shutdown."""
from pathlib import Path
import argparse,json,subprocess,os,sys,time,socket,threading,select
from dotenv import dotenv_values
import requests
p=argparse.ArgumentParser();p.add_argument('--env',required=True);p.add_argument('--accounts',required=True);a=p.parse_args();config=Path(a.env).resolve();v={k:x for k,x in dotenv_values(config).items() if x is not None};ROOT=Path(__file__).resolve().parents[1];O=Path(v['PAMK_EVIDENCE_DIR']);web=v['PAMK_WEB_PYTHON'];account=json.loads(Path(a.accounts).read_text())['animation'];logs=[];checks=[]
def service(action,envfile):subprocess.run([web,'-B',str(ROOT/'scripts/local_service.py'),action,'--env',str(envfile)],check=True,capture_output=True,text=True)
def variant(name,changes):
 d={**v,**changes};f=config.parent/f'{name}.env';f.write_text(''.join(k+'='+json.dumps(x,ensure_ascii=False)+'\n' for k,x in d.items()));f.chmod(0o600);return f

def login():
 s=requests.Session();s.trust_env=False
 for _ in range(80):
  try:r=s.get('http://127.0.0.1:8770/login/',timeout=2);break
  except requests.RequestException:time.sleep(.1)
 r=s.post('http://127.0.0.1:8770/login/',data={'username':account['username'],'password':account['password'],'csrfmiddlewaretoken':s.cookies['csrftoken']});assert r.status_code==200 and r.url.endswith('/recommend/');return s

def call(s,label,k=61):
 t=time.monotonic();r=s.post('http://127.0.0.1:8770/api/recommendations/',json={'top_k':k},headers={'X-CSRFToken':s.cookies['csrftoken']},timeout=180)
 try:body=r.json()
 except ValueError:body={'raw':r.text[:400]}
 logs.append({'case':label,'status':r.status_code,'http_seconds':time.monotonic()-t,'response':body});return r,body

def ck(name,x):checks.append({'case':name,'pass':bool(x)});print(('PASS ' if x else 'FAIL ')+name,flush=True)
# Test only a forwarding proxy connection, never stop/modify the real MySQL service.
blocked=threading.Event();closed=threading.Event();listener=socket.socket();listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(('127.0.0.1',13306));listener.listen();listener.settimeout(.2)
def relay(c):
 if blocked.is_set():c.close();return
 upstream=socket.create_connection(('127.0.0.1',3306));socks=[c,upstream]
 try:
  while not closed.is_set():
   ready,_,_=select.select(socks,[],[],.2)
   for s in ready:
    data=s.recv(65536)
    if not data:return
    (upstream if s is c else c).sendall(data)
 finally:c.close();upstream.close()
def serve():
 while not closed.is_set():
  try:c,_=listener.accept();threading.Thread(target=relay,args=(c,),daemon=True).start()
  except socket.timeout:pass
threading.Thread(target=serve,daemon=True).start()
try:
 service('stop',config)
 for label,changes in [('worker_missing',{'PAMK_WORKER_PYTHON':'/nonexistent/release-worker-python'}),('worker_timeout',{'PAMK_WORKER_TIMEOUT':'0.001'})]:
  env=variant(label,changes);service('start',env);s=login();r,b=call(s,label,62 if label=='worker_missing' else 63);ck(label+'_explicit_503',r.status_code==503 and b.get('error') in ['MODEL_WORKER_UNAVAILABLE','MODEL_INFERENCE_UNAVAILABLE']);service('stop',env)
 env=variant('db_proxy',{'DB_PORT':'13306'});service('start',env);s=login();blocked.set();r,b=call(s,'database_temporarily_unavailable',64);ck('database_error_503',r.status_code==503 and b.get('error')=='DATABASE_UNAVAILABLE');blocked.clear();r,b=call(s,'database_recovered',64);ck('database_recovers_real_model',r.status_code==200 and b.get('source')=='PAMK_HISTORY_MEAN_PROFILE');service('stop',env)
 env=variant('flag_off',{'PAMK_INDUCTIVE_ENABLED':'0'});service('start',env);s=login();r,b=call(s,'feature_flag_off',65);ck('rollback_cold_honest',r.status_code==200 and b.get('model_name') is None and b.get('source')=='METADATA_PREFERENCE');service('stop',env)
 service('start',config);s=login();r,b=call(s,'restored_default',66);ck('restored_mean_real_gpu',r.status_code==200 and b.get('model_name')=='PAMK-HistoryMean-v1' and b.get('model_invoked'))
finally:
 blocked.clear();closed.set();listener.close()
 (O/'fault_requests.json').write_text(json.dumps(logs,ensure_ascii=False,indent=2));(O/'fault_summary.json').write_text(json.dumps({'pass':sum(x['pass'] for x in checks),'fail':sum(not x['pass'] for x in checks),'checks':checks,'NEW_TEST_ACCESS':0,'oom_test':'NOT_RUN; real timeout used instead, no intentional VRAM exhaustion'},indent=2))
print(json.dumps(checks));assert all(x['pass'] for x in checks)
