"""PAMK-Inductive-v1: trained history encoder, frozen PAMK scoring.

No research user embedding lookup. No scientific source modification. Arbitrary
history context uses the same target-exclusion/channel formulas as PAMK.
"""
from __future__ import annotations
import math
import torch
from torch import nn

VERSION = 'PAMK-Inductive-v1'

class HistoryEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.LayerNorm(input_dim),nn.Linear(input_dim,hidden),nn.GELU(),nn.Linear(hidden,output_dim))
    def forward(self,x):return self.net(x)

class InductivePAMK(nn.Module):
    def __init__(self, pamk):
        super().__init__()
        self.pamk=pamk.eval().requires_grad_(False)
        p=pamk.parent
        # All four banks are frozen movie features, never other users' vectors.
        with torch.no_grad():
            bank=torch.cat((p.item_embedding.weight[:9031],pamk._eval_text_cache,pamk._eval_image_cache,pamk._eval_kg_cache),dim=-1).detach()
        self.register_buffer('history_bank',bank,persistent=False)
        self.encoder=HistoryEncoder(bank.shape[1]+1,p.embedding_dim).to(bank.device)
    def representation(self, history, weights=None):
        mask=history>=0;ids=history.clamp_min(0)
        w=mask.float() if weights is None else mask.float()*weights
        denom=w.sum(1,keepdim=True).clamp_min(1)
        pooled=(self.history_bank[ids]*w.unsqueeze(-1)).sum(1)/denom
        feature=torch.cat((pooled,torch.log1p(mask.sum(1,keepdim=True).float())/math.log(501)),dim=-1)
        return self.encoder(feature)
    def context(self,history,targets):
        c=self.pamk.context;batch,n=targets.shape;h=history.clamp_min(0)
        valid=(history[:,None,:]>=0)&(h[:,None,:]!=targets[:,:,None])
        mask=valid.float();den=mask.sum(-1).clamp_min(1);cnt=mask.sum(-1)
        k=c.kernels[:,targets[:,:,None],h[:,None,:]].permute(1,2,3,0)
        m=c.mm_sim[:,targets[:,:,None],h[:,None,:]].permute(1,2,3,0)
        k=k*mask[:,:,:,None];m=m*mask[:,:,:,None]
        raw=k.sum(2)/den[:,:,None];mm=m.sum(2)/den[:,:,None]
        a=(cnt/(cnt+c.shrinkage_tau0))[:,:,None]*(raw-c.bg[targets])
        cv=((k[:,:,:,:,None]*m[:,:,:,None,:]).sum(2)/den[:,:,None,None]-raw[:,:,:,None]*mm[:,:,None,:]).reshape(batch,n,8)
        pop=c.item_pop[h][:,None,:]*mask
        mu=pop.sum(-1)/den;var=(pop.square().sum(-1)/den-mu.square()).clamp_min(0)
        pt=c.item_pop[targets];nu=c.item_niche[targets]
        freq=torch.stack((pt,nu,mu,torch.sqrt(var+1e-8),pt-mu,torch.log1p(cnt)/math.log(501)),dim=-1)
        az,cz,mz=c.standardize_channels(a,cv,mm)
        values=dict(a=a,cv=cv,m_mean=mm,a_std=az,cv_std=cz,m_std=mz,freq_ctx=freq,p_t=pt,nu_t=nu,valid_cnt=cnt)
        return {key:value.reshape(batch*n,*value.shape[2:]) for key,value in values.items()}
    def parent_features(self,u,items,history):
        p=self.pamk.parent;batch,n=items.shape;mask=history>=0
        keys0,_,pool0=p._history(history,mask)
        user=u[:,None,:].expand(batch,n,-1)
        item=p.item_embedding(items);semantic=p._text(items);genre=p._genre(items)
        keys=keys0[:,None,:,:].expand(-1,n,-1,-1);hm=mask[:,None,:].expand(-1,n,-1)
        hist_pool=pool0[:,None,:].expand(batch,n,-1);hist_attn=p._attention_pool(item,keys,hm)
        flat=lambda x:x.reshape(batch*n,x.shape[-1])
        uf,iv,gv,hp,ha,sv=map(flat,(user,item,genre,hist_pool,hist_attn,semantic))
        shared=torch.cat((uf,iv,gv,hp,sv),-1)
        zsk=p.mlp_sk(shared);zbh=p.mlp_behavior(torch.cat((shared,ha),-1))
        zq,_,_,_=p.grouped_quadratic(uf,iv,torch.cat((hp,ha),-1),torch.cat((sv,gv),-1))
        q,b,_,_,_,_=p.cross_scale_balance(zq,zbh)
        hif=p.hybrid_fusion(zbh,zq,q*b)[0];gate=p.vector_gate(zsk)
        hui=gate*hif+(1-gate)*zsk
        score=p.dnn_predict(hui).squeeze(-1)+p.semantic_linear(sv).squeeze(-1)
        return score,hui,hif,sv
    def forward(self,history,items,weights=None,mode='learned',telemetry=False,user_vector=None):
        if user_vector is not None:u=user_vector
        elif mode=='learned':u=self.representation(history,weights)
        elif mode=='mean':
            mask=(history>=0).float();u=(self.pamk.parent.item_embedding(history.clamp_min(0))*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        else:raise ValueError('INVALID_ENCODER_MODE')
        ps,hui,hif,semantic=self.parent_features(u,items,history)
        ctx=self.context(history,items)
        delta,t=self.pamk._compute_delta_from_features(items.reshape(-1),hui,hif,semantic,ctx,return_telemetry=telemetry)
        scores=(ps+self.pamk.parent.dnn_predict(delta).squeeze(-1)).reshape(items.shape)
        return (scores,u,t) if telemetry else scores
