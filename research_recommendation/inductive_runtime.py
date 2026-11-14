"""Lazy adapter in the same single-GPU worker as native PAMK."""
from pathlib import Path
from collections import OrderedDict
import hashlib,json,os,time,uuid
from inductive_model import InductivePAMK,VERSION
from encoding_policy import policy

class InductiveRuntime:
    def __init__(self,engine):
        self.engine=engine;self.representations=OrderedDict();self.encoder_calls=0;self.representation_calls=0
        path=Path(os.environ['PAMK_INDUCTIVE_MANIFEST']);self.manifest=json.loads(path.read_text())
        from pamk_worker import sha,PAMK_FULL_SHA
        m=self.manifest
        if m['version']!=VERSION or m['base_sha256']!=PAMK_FULL_SHA or sha(Path(m['checkpoint']))!=m['sha256'] or sha(Path(__file__).with_name('inductive_model.py'))!=m['encoder_source_sha256']:
            raise RuntimeError('INDUCTIVE_IDENTITY_MISMATCH')
        torch=engine.torch
        saved=torch.load(m['checkpoint'],map_location=engine.device,weights_only=False)
        if saved['config']['base_checkpoint_sha256']!=PAMK_FULL_SHA:raise RuntimeError('INDUCTIVE_BASE_MISMATCH')
        self.model=InductivePAMK(engine.model);self.model.encoder.load_state_dict(saved['encoder']);self.model.encoder.eval().requires_grad_(False)
    def infer(self,request):
        e=self.engine;np,torch=e.np,e.torch;started=time.monotonic()
        h=request.get('history');excluded=request.get('exclude_canonical_ids',[]);k=request['top_k'];account=request.get('user')
        if not isinstance(account,str) or not account.startswith('business:') or not account[9:].isdecimal():raise ValueError('INVALID_ACCOUNT_SCOPE')
        if type(h) is not list or not 1<=len(h)<=20 or any(type(i) is not int or not 0<=i<9031 for i in h) or len(set(h))!=len(h):raise ValueError('INVALID_HISTORY')
        if type(excluded) is not list or len(excluded)>9031 or any(type(i) is not int or not 0<=i<9031 for i in excluded):raise ValueError('INVALID_EXCLUSIONS')
        method=request.get('encoding_method','learned');chosen=policy(method)
        if request.get('adapter_identity')!=chosen['encoding_identity']:raise RuntimeError('ADAPTER_VERSION_CHANGED')
        context=request.get('context_identity')
        if not isinstance(context,str) or len(context)!=64:raise ValueError('INVALID_CONTEXT_IDENTITY')
        key=(account,context,chosen['encoding_identity'],tuple(h))
        ht=torch.full((1,20),-1,device=e.device,dtype=torch.long);ht[0,:len(h)]=torch.tensor(h,device=e.device)
        hit=key in self.representations
        with torch.inference_mode():
            if hit:vector=self.representations.pop(key)
            else:
                self.representation_calls+=1
                if method=='learned':vector=self.model.representation(ht);self.encoder_calls+=1
                else:
                    mask=(ht>=0).float()
                    vector=(e.model.parent.item_embedding(ht.clamp_min(0))*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            self.representations[key]=vector
            while len(self.representations)>128:self.representations.popitem(last=False)
            begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record()
            parts=[];telemetry={x:[] for x in ['pi_text','pi_vis','pi_kg']}
            for j in range(0,9031,1024):
                items=torch.arange(j,min(j+1024,9031),device=e.device)[None,:]
                score,_,t=self.model(ht,items,user_vector=vector,telemetry=True);parts.append(score.cpu().numpy().reshape(-1))
                for x in telemetry:telemetry[x].extend(t[x].cpu().numpy().reshape(-1).tolist())
            end.record();torch.cuda.synchronize();gpu_ms=begin.elapsed_time(end)
        scores=np.concatenate(parts);assert np.isfinite(scores).all()
        seen=set(excluded)|set(h);order=np.argsort(-scores,kind='stable');eligible=[int(i) for i in order if int(i) not in seen]
        items=[{'canonical_movie_id':str(cid),'rank':rank,'original_model_rank':int(np.flatnonzero(order==cid)[0])+1,'score':float(scores[cid]),'telemetry':{x:float(v[cid]) for x,v in telemetry.items()}} for rank,cid in enumerate(eligible[:k],1)]
        e.inference_count+=1
        vector_sha=hashlib.sha256(vector.cpu().numpy().tobytes()).hexdigest()
        result={'receipt_id':str(uuid.uuid4()),'model_name':chosen['model_name'],'checkpoint_identity':self.manifest['base_sha256'],'adapter_checkpoint_sha256':chosen['encoder_checkpoint_sha256'],'adapter_version':chosen['version'],'encoding_method':method,'encoding_identity':chosen['encoding_identity'],'representation_compute_count':self.representation_calls,'user':account,'context_identity':context,'user_vector_sha256':vector_sha,'history_count':len(h),'representation_cache_hit':hit,'encoder_call_count':self.encoder_calls,'inference_count':e.inference_count,'inference_mode':'REAL_ONLINE_INFERENCE','smoke':False,'items':items,'catalogue_size':9031,'candidate_count':9031-len(seen),'score_digest':hashlib.sha256(scores.tobytes()).hexdigest(),'device':str(e.device),'worker_pid':os.getpid(),'score_seconds':time.monotonic()-started,'gpu_forward_ms':gpu_ms,'load_seconds':e.load_seconds,'vram_peak_allocated_mib':torch.cuda.max_memory_allocated()/(1<<20),'NEW_TEST_ACCESS':0}
        Path(os.environ['PAMK_EVIDENCE_DIR'],f'INDUCTIVE_RECEIPT_{result["receipt_id"]}.json').write_text(json.dumps(result))
        return result
