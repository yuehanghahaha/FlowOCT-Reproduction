"""Local capacity diagnostic: one fixed real B-scan, t=0, velocity MSE only.

This is not the original FlowOCT training objective or a reproduction score.
Core model, Haar and data loader are imported without editing them.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from skimage.metrics import structural_similarity

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'flowoct_reproduction'))
from flowoct.data import read_manifest
from flowoct.runner import setup,make_model,environment


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_json(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def main():
    HERE.mkdir(exist_ok=True)
    if (HERE/'result.json').exists():raise FileExistsError('Completed control already exists')
    ref=ROOT/'analysis/flowoct_singlecase_results_20261026/raw_review'
    protocol=json.loads((ref/'protocol.json').read_text(encoding='utf-8'))
    source={k:sha(ROOT/'flowoct_reproduction'/k) for k in protocol['baseline']['source_sha256']}
    assert source==protocol['baseline']['source_sha256']
    cfg=dict(protocol['config'],device='cuda:0',manifest=str(ROOT/'data/OCTA500_common_v1/split_3M_z256.csv'))
    setup(cfg)
    rows=read_manifest(cfg['manifest'],'3M');case=next(r for r in rows if r['case_id']=='10308')
    helper=ROOT/'FlowOCT_singlecase_tools/FlowOCT_singlecase.py'
    import importlib.util
    spec=importlib.util.spec_from_file_location('singlecase_helper',helper)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    item=mod.load_fixed_item([case],cfg)
    proof=mod.item_proof(item)
    assert proof==json.loads((ref/'fixed_input.json').read_text(encoding='utf-8'))
    plane_y=128
    a=item['A'][:,plane_y].unsqueeze(0).to('cuda:0')
    b=item['B'][:,plane_y].unsqueeze(0).to('cuda:0')
    gt=b[0,0].cpu().numpy().T.copy()
    a_image=a[0,0].cpu().numpy().T.copy()
    model=make_model(cfg,False)
    initial_hash=mod.model_hash(model.netG.state_dict())
    optimizer=torch.optim.Adam(model.netG.parameters(),lr=cfg['lr'],betas=(cfg['beta'],.999))
    t0=torch.zeros(1,device='cuda:0');t1=torch.ones(1,device='cuda:0')
    records=[];steps=(0,25,50,100,200,400,800,1200,1600,2000)
    torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter()
    write_json(HERE/'protocol.json',{'case_id':'10308','fixed_y':plane_y,'seed':cfg['seed'],
        'preprocessing_and_input_hashes':proof,'source_sha256':source,
        'initialization':'fresh random; no pretrained checkpoint',
        'input':'one fixed OCT B-scan256x256; t=0; no GT in input',
        'model':'unchanged2D FlowOCT netG plus original Haar/euler_step',
        'objective':'10*MSE(A+v_hat(A,t0), B); exactly equivalent to10*MSE(v_hat,B-A)',
        'differences':'No random slices or t, no L1/contrast/projection losses, no ROI, no clamping before training loss; batch1',
        'optimizer':{'lr':cfg['lr'],'betas':[cfg['beta'],.999]},'total_updates':2000,
        'limits':'Local6GiB GPU; einops0.8.2 vs formal0.8.1; a memory positive control, not formal baseline or a causal full-pipeline ablation',
        'environment':environment(torch.device('cuda:0'))})
    for step in range(2001):
        grad_norm=None
        if step:
            model.train();optimizer.zero_grad(set_to_none=True)
            raw=model.euler_step(a,t0,t1)
            loss=10*torch.mean((raw-b)**2)
            if not torch.isfinite(loss):raise FloatingPointError('Non-finite control loss')
            loss.backward()
            if step in steps:
                grad_norm=float(torch.sqrt(sum(torch.sum(p.grad**2) for p in model.netG.parameters() if p.grad is not None)))
            optimizer.step()
        if step in steps:
            model.eval()
            with torch.no_grad():pred=model.euler_step(a,t0,t1)[0,0].cpu().numpy().T.copy()
            clipped=np.clip(pred,0,1)
            diff=pred.astype(np.float64)-gt
            bright=gt>.05
            score={'step':step,'raw_mse':float(np.mean(diff**2)),
                   'raw_mae':float(np.mean(abs(diff))),
                   'clipped_ssim_range1':float(structural_similarity(gt,clipped,data_range=1.,win_size=7)),
                   'bright_gt_threshold_diagnostic':.05,'bright_gt_fraction':float(bright.mean()),
                   'bright_gt_raw_mae':float(abs(diff[bright]).mean()),
                   'gradient_l2_norm':grad_norm,'elapsed_seconds':time.perf_counter()-start,
                   'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            records.append(score)
            np.savez_compressed(HERE/f'step{step:04d}.npz',oct=a_image,gt=gt,prediction_raw=pred,prediction_clipped=clipped)
            write_json(HERE/'progress.json',{'records':records})
            print(json.dumps(score,ensure_ascii=False),flush=True)
        elif step%100==0:print(f'CONTROL step{step}/2000',flush=True)
    final_hash=mod.model_hash(model.netG.state_dict())
    assert initial_hash!=final_hash
    assert source=={k:sha(ROOT/'flowoct_reproduction'/k) for k in source}
    assert {float(s['step']) for s in optimizer.state.values()}=={2000.}
    torch.save({'format':'fixed-plane-memory-control-v1','model':model.state_dict(),'steps':2000,'protocol':json.loads((HERE/'protocol.json').read_text(encoding='utf-8'))},HERE/'final_control.pt')
    result={'records':records,'initial_netg_hash':initial_hash,'final_netg_hash':final_hash,
            'core_unchanged':True,'original_input_exactly_matches':True,'actual_adam_updates':2000,
            'warning':'Positive control with simplified objective; not full FlowOCT reproduction or proof of whole-volume fitting'}
    write_json(HERE/'result.json',result)
    print('COMPLETE',flush=True)


if __name__=='__main__':main()
