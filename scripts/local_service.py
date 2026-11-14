#!/usr/bin/env python3
"""Explicit local acceptance lifecycle; credentials never appear on command lines."""
from pathlib import Path
import argparse,os,sys,json,subprocess,signal,time,hashlib
from dotenv import dotenv_values
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('action',choices=['start','run','stop','status','check','migrate','init-db','seed-from']);p.add_argument('--env',required=True);p.add_argument('--port',type=int,default=8770);p.add_argument('--source-db');a=p.parse_args()
config=Path(a.env).resolve();settings={k:v for k,v in dotenv_values(config).items() if v is not None};env={**os.environ,**settings,'DJANGO_SETTINGS_MODULE':'movie.release_settings','NEW_TEST_ACCESS':'0','PYTHONDONTWRITEBYTECODE':'1'};db=env.get('PAMK_DATABASE_NAME','')
if not db.startswith('ch5_pamk_') or not db.replace('_','').isalnum():raise SystemExit('ISOLATED_DATABASE_REQUIRED')
runtime=config.parent;runtime.mkdir(parents=True,exist_ok=True);state=runtime/f'service-{a.port}.json';log=runtime/f'service-{a.port}.log';web=env['PAMK_WEB_PYTHON']
def readstate():return json.loads(state.read_text()) if state.exists() else {}
def valid(info):
 pid=info.get('pid');q=Path('/proc')/str(pid)
 try:return q.exists() and str(Path(info['source'])/'scripts/local_service.py') in (q/'cmdline').read_text().replace('\0',' ') and (q/'stat').read_text().split()[21]==info['start_ticks']
 except (OSError,KeyError):return False
def mysql():
 import MySQLdb
 return MySQLdb.connect(host=env.get('DB_HOST','127.0.0.1'),user=env['DB_USER'],passwd=env['DB_PASSWORD'],port=int(env.get('DB_PORT','3306')))
def manage(*args):return subprocess.run([web,'-B',str(ROOT/'manage.py'),*args],env=env,cwd=ROOT,check=True)
if a.action=='init-db':
 c=mysql()
 with c.cursor() as q:q.execute(f'CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4')
 c.close();print('ISOLATED_DATABASE_READY',db)
elif a.action=='seed-from':
 source=a.source_db or ''
 if not source.startswith('ch5_pamk_') or not source.replace('_','').isalnum() or source==db:raise SystemExit('ISOLATED_SOURCE_REQUIRED')
 c=mysql()
 with c.cursor() as q:
  q.execute('SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=%s',[db])
  if q.fetchone()[0]:raise SystemExit('TARGET_NOT_EMPTY_NO_OVERWRITE')
 c.close();childenv=dict(env,MYSQL_PWD=env['DB_PASSWORD']);base=['-h',env.get('DB_HOST','127.0.0.1'),'-P',env.get('DB_PORT','3306'),'-u',env['DB_USER']]
 dump=subprocess.Popen(['mysqldump',*base,'--single-transaction','--no-tablespaces','--skip-comments',source],stdout=subprocess.PIPE,env=childenv)
 load=subprocess.run(['mysql',*base,db],stdin=dump.stdout,env=childenv);dump.stdout.close();code=dump.wait()
 if code or load.returncode:raise SystemExit('ISOLATED_COPY_FAILED_REVIEW_PARTIAL_TARGET')
 print('ISOLATED_COPY_COMPLETE',source,'->',db)
elif a.action=='check':
 m=json.loads(Path(env['PAMK_ASSET_CONFIG']).read_text());items=dict(m['assets'],checkpoint=m['checkpoint']);encoder=json.loads(Path(env['PAMK_INDUCTIVE_MANIFEST']).read_text());items['inductive_encoder']={'path':encoder['checkpoint'],'sha256':encoder['sha256']}
 for key,x in items.items():
  path=Path(x['path']);h=hashlib.sha256()
  if not path.is_file():raise SystemExit('ASSET_MISSING:'+key+':'+str(path))
  with path.open('rb') as f:
   for b in iter(lambda:f.read(8<<20),b''):h.update(b)
  if h.hexdigest()!=x['sha256']:raise SystemExit('ASSET_SHA_MISMATCH:'+key)
  print('VERIFIED',key,x['sha256'])
 manage('check');manage('migrate','--check')
elif a.action=='migrate':manage('migrate','--noinput')
elif a.action=='status':
 info=readstate();print(json.dumps({'running':valid(info),'state':info,'log':str(log)},ensure_ascii=False))
elif a.action=='stop':
 info=readstate()
 if valid(info):
  os.kill(info['pid'],signal.SIGTERM)
  for _ in range(100):
   if not valid(info):break
   time.sleep(.1)
  if valid(info):raise SystemExit('GRACEFUL_STOP_TIMEOUT')
 print('STOPPED_OR_NOT_RUNNING')
elif a.action=='start':
 if valid(readstate()):raise SystemExit('ALREADY_RUNNING')
 with log.open('ab') as f:proc=subprocess.Popen([web,'-B',str(Path(__file__).resolve()),'run','--env',str(config),'--port',str(a.port)],stdout=f,stderr=f,env=env,cwd=ROOT,start_new_session=True)
 time.sleep(1)
 if proc.poll() is not None:raise SystemExit('START_FAILED_SEE_LOG:'+str(log))
 print('STARTED',proc.pid,'log',log)
else:
 if valid(readstate()):raise SystemExit('ALREADY_RUNNING')
 import socket
 with socket.socket() as probe:
  probe.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
  try:probe.bind(('127.0.0.1',a.port))
  except OSError:raise SystemExit('PORT_BUSY_STATE_NOT_CHANGED')
 pid=os.getpid();state.write_text(json.dumps({'pid':pid,'start_ticks':Path('/proc',str(pid),'stat').read_text().split()[21],'source':str(ROOT),'port':a.port,'database':db,'strategy':env.get('PAMK_INDUCTIVE_STRATEGY','mean')},indent=2));state.chmod(0o600)
 proc=subprocess.Popen([web,'-B',str(ROOT/'manage.py'),'runserver',f'127.0.0.1:{a.port}','--noreload'],env=env,cwd=ROOT)
 def stop(*_):raise KeyboardInterrupt
 signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
 try:raise SystemExit(proc.wait())
 except KeyboardInterrupt:pass
 finally:
  # Only terminate this Django process's dedicated inference children.
  children=[]
  for q in Path('/proc').iterdir():
   if not q.name.isdecimal():continue
   try:
    if int((q/'status').read_text().split('PPid:')[1].splitlines()[0])==proc.pid and 'pamk_worker.py' in (q/'cmdline').read_text():children.append(int(q.name))
   except (OSError,ValueError):pass
  for child in children:
   try:os.kill(child,signal.SIGTERM)
   except ProcessLookupError:pass
  if proc.poll() is None:proc.terminate()
  try:proc.wait(timeout=10)
  except subprocess.TimeoutExpired:proc.kill();proc.wait()
