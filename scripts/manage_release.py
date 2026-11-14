#!/usr/bin/env python3
from pathlib import Path
import argparse,os,subprocess
from dotenv import dotenv_values
p=argparse.ArgumentParser();p.add_argument('--env',required=True);p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args();e={**os.environ,**{k:v for k,v in dotenv_values(a.env).items() if v is not None},'DJANGO_SETTINGS_MODULE':'movie.release_settings','NEW_TEST_ACCESS':'0','PYTHONDONTWRITEBYTECODE':'1'};root=Path(__file__).resolve().parents[1]
raise SystemExit(subprocess.call([e['PAMK_WEB_PYTHON'],'-B',str(root/'manage.py'),*a.args],cwd=root,env=e))
