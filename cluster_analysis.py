#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ClusterAnalysis_T229.py
Run separately for L=32, 64, 128 by changing only L below.
Loads the existing EnhancedInvNet checkpoint; no retraining.
Computes periodic geometric spin-cluster statistics near T=2.29:
 f_max = largest-cluster fraction
 S_cl = sum(s^2)/sum(s), excluding one largest cluster per configuration
 P_span = fraction of configurations with a wrapping cluster on the torus
"""
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

L = 128  # CHANGE ONLY THIS: 32, 64, 128
TARGET_T = 2.29
SEED = 123
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = (SCRIPT_DIR / "../JOB5_Noise/J5Data").resolve()
CLEAN_CSV = DATA_DIR / f"MCD{L}.csv"
NOISY_CSV = DATA_DIR / f"MCDN{L}.csv"
CHECKPOINT = Path(f"/home/partha02965/JOB_SCRIPTA/Generated_L{L}/best_model.pth")
OUT_DIR = SCRIPT_DIR / f"ClusterAnalysis_L{L}_T229"
BATCH_SIZE = {32:64, 64:16, 128:2}[L]
CHUNK = {32:4096, 64:1024, 128:128}[L]

def seed_all():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

def spin_cols(path):
    cols = pd.read_csv(path,nrows=0).columns
    s=[c for c in cols if c.lower().startswith("spin")]
    s.sort(key=lambda x:int("".join(filter(str.isdigit,x)) or 0))
    if len(s)!=L*L: raise ValueError(f"Expected {L*L} spin columns, found {len(s)}")
    return s

def test_indices(n):
    nt=int(.8*n); nv=int(.1*n)
    p=torch.randperm(n,generator=torch.Generator().manual_seed(SEED)).tolist()
    return p[nt+nv:]

def selected_rows(path, indices, usecols, dtype):
    wanted=set(indices); pieces=[]; off=0
    for ch in pd.read_csv(path,usecols=usecols,dtype=dtype,chunksize=CHUNK):
        pos=sorted(i-off for i in wanted if off<=i<off+len(ch))
        if pos:
            q=ch.iloc[pos].copy(); q["_idx"]=[off+j for j in pos]; pieces.append(q)
        off+=len(ch)
    d=pd.concat(pieces,ignore_index=True)
    order={v:k for k,v in enumerate(indices)}
    d["_ord"]=d["_idx"].map(order)
    return d.sort_values("_ord").drop(columns="_ord").reset_index(drop=True)

def load_data():
    sc=spin_cols(CLEAN_CSV)
    n=len(pd.read_csv(CLEAN_CSV,usecols=["Temperature"]))
    idx=test_indices(n)
    dc={c:np.int8 for c in sc}; dc["Temperature"]=np.float32
    dn={c:np.int8 for c in sc}
    c=selected_rows(CLEAN_CSV,idx,["Temperature"]+sc,dc)
    y=selected_rows(NOISY_CSV,idx,sc,dn)
    if not np.array_equal(c["_idx"],y["_idx"]): raise RuntimeError("Rows not aligned")
    t=c["Temperature"].to_numpy(np.float32)
    grid=np.unique(np.round(t.astype(float),6))
    actual=float(grid[np.argmin(abs(grid-TARGET_T))])
    keep=np.isclose(t,actual,atol=1e-5)
    clean=c.loc[keep,sc].to_numpy(np.float32).reshape(-1,L,L)
    noisy=y.loc[keep,sc].to_numpy(np.float32).reshape(-1,L,L)
    ids=c.loc[keep,"_idx"].to_numpy(np.int64)
    mask=(noisy!=0).astype(np.float32)  # identical to existing trained model
    x=np.stack([noisy,mask],axis=1).astype(np.float32)
    print(f"L={L}: N={len(clean)}, actual grid T={actual:.6f}")
    return x,clean,ids,actual

class DS(Dataset):
    def __init__(self,x,s,i):
        self.x=torch.from_numpy(x); self.s=torch.from_numpy(s)[:,None]; self.i=torch.from_numpy(i)
    def __len__(self): return len(self.x)
    def __getitem__(self,k): return self.x[k],self.s[k],self.i[k]

class EnhancedInvNet(nn.Module):
    def __init__(self):
        super().__init__(); C=64
        self.enc_conv1=nn.Conv2d(2,C,3,padding=1,padding_mode="circular")
        self.enc_conv2=nn.Conv2d(C,C,3,padding=1,padding_mode="circular")
        self.enc_conv3=nn.Conv2d(C,C,3,padding=1,padding_mode="circular")
        self.enc_conv4=nn.Conv2d(C,C,3,padding=1,padding_mode="circular")
        self.enc_conv5=nn.Conv2d(C,C,3,padding=1,padding_mode="circular")
        self.attention=nn.Sequential(nn.Conv2d(C,C//4,1),nn.ReLU(inplace=True),nn.Conv2d(C//4,1,1),nn.Sigmoid())
        self.dec_conv1=nn.Conv2d(C+1,64,3,padding=1,padding_mode="circular")
        self.dec_conv2=nn.Conv2d(64,64,3,padding=1,padding_mode="circular")
        self.dec_conv3=nn.Conv2d(64,32,3,padding=1,padding_mode="circular")
        self.dec_conv4=nn.Conv2d(32,16,3,padding=1,padding_mode="circular")
        self.dec_conv5=nn.Conv2d(16,1,1)
        self.head_T=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(C,128),nn.ReLU(inplace=True),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(inplace=True),nn.Linear(64,1))
        self.head_P=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(C,128),nn.ReLU(inplace=True),nn.Dropout(.1),nn.Linear(128,64),nn.ReLU(inplace=True),nn.Linear(64,2))
        self.a=nn.ReLU(inplace=True)
    def forward(self,x):
        noisy=x[:,:1]; mask=x[:,1:2]
        h1=self.a(self.enc_conv1(x)); h2=self.a(self.enc_conv2(h1))+h1
        h3=self.a(self.enc_conv3(h2))+h2; h4=self.a(self.enc_conv4(h3))+h3
        h5=self.a(self.enc_conv5(h4))+h4; A=self.attention(h5)
        d1=self.a(self.dec_conv1(torch.cat([h5,A],1))); d2=self.a(self.dec_conv2(d1))+d1
        d3=self.a(self.dec_conv3(d2)); d4=self.a(self.dec_conv4(d3))
        raw=torch.tanh(self.dec_conv5(d4))
        rec=mask*noisy+(1-mask)*raw
        return rec,self.head_T(h5),self.head_P(h5),A

def model_load(device):
    m=EnhancedInvNet().to(device)
    try: ck=torch.load(CHECKPOINT,map_location=device,weights_only=False)
    except TypeError: ck=torch.load(CHECKPOINT,map_location=device)
    sd=ck["model_state_dict"] if isinstance(ck,dict) and "model_state_dict" in ck else ck
    sd={k.removeprefix("module."):v for k,v in sd.items()}
    m.load_state_dict(sd,strict=True); m.eval(); return m

@torch.no_grad()
def reconstruct(m,loader,device):
    cc=[]; rr=[]; ii=[]
    for x,s,idx in loader:
        r,*_=m(x.to(device))
        rb=torch.where(r>0,torch.ones_like(r),-torch.ones_like(r))
        cc.append(s[:,0].numpy().astype(np.int8))
        rr.append(rb[:,0].cpu().numpy().astype(np.int8)); ii.append(idx.numpy())
    return np.concatenate(cc),np.concatenate(rr),np.concatenate(ii)

class UF:
    def __init__(self,n):
        self.p=np.arange(n); self.sz=np.ones(n,dtype=int)
        self.d=np.zeros((n,2),dtype=int); self.wx=np.zeros(n,bool); self.wy=np.zeros(n,bool)
    def find(self,x):
        if self.p[x]==x:return x,np.zeros(2,dtype=int)
        p=int(self.p[x]); r,dp=self.find(p); dx=self.d[x]+dp
        self.p[x]=r; self.d[x]=dx; return r,dx.copy()
    def union(self,a,b,delta):
        ra,da=self.find(a); rb,db=self.find(b)
        if ra==rb:
            cyc=(db-da)-delta
            if cyc[0]!=0:self.wx[ra]=True
            if cyc[1]!=0:self.wy[ra]=True
            return
        if self.sz[ra]<self.sz[rb]:
            self.p[ra]=rb; self.d[ra]=db-da-delta; self.sz[rb]+=self.sz[ra]
            self.wx[rb]|=self.wx[ra]; self.wy[rb]|=self.wy[ra]
        else:
            self.p[rb]=ra; self.d[rb]=da+delta-db; self.sz[ra]+=self.sz[rb]
            self.wx[ra]|=self.wx[rb]; self.wy[ra]|=self.wy[rb]

def metrics(s):
    uf=UF(L*L); ix=lambda r,c:r*L+c
    for r in range(L):
        for c in range(L):
            a=ix(r,c); c2=(c+1)%L; r2=(r+1)%L
            if s[r,c]==s[r,c2]:uf.union(a,ix(r,c2),np.array([0,1]))
            if s[r,c]==s[r2,c]:uf.union(a,ix(r2,c),np.array([1,0]))
    roots=np.array([uf.find(i)[0] for i in range(L*L)])
    ur,cnt=np.unique(roots,return_counts=True); cnt=cnt.astype(float)
    j=int(np.argmax(cnt)); fmax=cnt[j]/(L*L); finite=np.delete(cnt,j)
    scl=float((finite**2).sum()/finite.sum()) if finite.size and finite.sum()>0 else 0.
    span=False
    for r in ur:
        q,_=uf.find(int(r)); span|=bool(uf.wx[q] or uf.wy[q])
    return fmax,scl,float(span),len(ur)

def ensemble(arr,ids,name,T):
    rows=[]
    for k,(s,i) in enumerate(zip(arr,ids),1):
        f,sc,p,nc=metrics(s)
        rows.append([L,T,name,int(i),f,sc,p,nc])
        if k%25==0 or k==len(arr):print(f"{name}: {k}/{len(arr)}")
    d=pd.DataFrame(rows,columns=["L","Temperature","Configuration","DatasetIndex","f_max","S_cl","spanning","N_clusters"])
    return d,{"L":L,"Temperature":T,"Configuration":name,"N":len(d),
              "f_max":d.f_max.mean(),"S_cl":d.S_cl.mean(),"P_span":d.spanning.mean()}

def save_tex(summary,path):
    with open(path,"w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\caption{Finite-size cluster characteristics of the clean and reconstructed configurations at $T\\approx2.29$.}\n")
        f.write("\\label{tab:cluster_analysis}\n\\begin{tabular}{c c c c c}\n\\hline\n")
        f.write("$L$ & Configuration & $f_{\\max}$ & $S_{\\mathrm{cl}}$ & $P_{\\mathrm{span}}$ \\\\\n\\hline\n")
        for _,r in summary.iterrows():
            f.write(f"{int(r.L)} & {r.Configuration} & {r.f_max:.3f} & {r.S_cl:.3f} & {r.P_span:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")

def main():
    seed_all()
    for p in [CLEAN_CSV,NOISY_CSV,CHECKPOINT]:
        if not p.exists():raise FileNotFoundError(p)
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:",device)
    x,s,ids,T=load_data()
    loader=DataLoader(DS(x,s,ids),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=device.type=="cuda")
    m=model_load(device); clean,recon,ids=reconstruct(m,loader,device)
    dc,sc=ensemble(clean,ids,"Clean",T); dr,sr=ensemble(recon,ids,"Reconstructed",T)
    per=pd.concat([dc,dr],ignore_index=True); summary=pd.DataFrame([sc,sr])
    per.to_csv(OUT_DIR/f"cluster_per_sample_L{L}_T229.csv",index=False)
    summary.to_csv(OUT_DIR/f"cluster_summary_L{L}_T229.csv",index=False)
    save_tex(summary,OUT_DIR/f"cluster_table_L{L}_T229.tex")
    print("\n",summary[["L","Temperature","Configuration","N","f_max","S_cl","P_span"]].to_string(index=False))
    print("\nOutputs:",OUT_DIR)

if __name__=="__main__":
    main()
