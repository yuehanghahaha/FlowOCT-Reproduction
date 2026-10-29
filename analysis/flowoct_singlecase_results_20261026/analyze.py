"""Independent analysis of downloaded2D maps and400-step history; no training."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
RAW=HERE/'raw_review'
ROOT=HERE.parents[1]
FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
FONT=font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':FONT.get_name(),'axes.unicode_minus':False,'font.size':10})


def js(path): return json.loads(path.read_text(encoding='utf-8-sig'))
def savejs(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def corr(a,b):
    a=a.astype(np.float64)-a.mean();b=b.astype(np.float64)-b.mean()
    return float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))
def scores(gt,fake):
    error=fake.astype(np.float64)-gt
    mse=max(float(np.mean(error**2)),np.finfo(float).eps)
    hgt=gt-gaussian_filter(gt,1);hfake=fake-gaussian_filter(fake,1)
    return {'mae':float(np.mean(abs(error))),'psnr_range1':float(-10*np.log10(mse)),
            'ssim_range1':float(structural_similarity(gt,fake,data_range=1.,win_size=7)),
            'ssim_gt_span':float(structural_similarity(gt,fake,data_range=float(np.ptp(gt)),win_size=7)),
            'highpass_correlation':corr(hgt,hfake),'lowpass_correlation':corr(gaussian_filter(gt,3),gaussian_filter(fake,3)),
            'highpass_std_ratio':float(hfake.std()/hgt.std()),'bias':float(error.mean()),
            'gradient_display_x_corr':corr(np.diff(gt,axis=1),np.diff(fake,axis=1)),
            'gradient_display_y_corr':corr(np.diff(gt,axis=0),np.diff(fake,axis=0)),
            'gt_min':float(gt.min()),'gt_max':float(gt.max()),'gt_span':float(np.ptp(gt))}


def main():
    stages=['initial_epoch0','epoch70','epoch100','epoch200','epoch300','best_fit','epoch400']
    history=[{k:float(v) for k,v in r.items()} for r in csv.DictReader((RAW/'training_history.csv').open(encoding='utf-8-sig'))]
    assert [int(r['epoch']) for r in history]==list(range(1,401))
    assert all(r['global_step']==r['epoch']==r['case_exposures'] for r in history)
    assert all(np.isfinite(v) for r in history for v in r.values())
    logs={int(m[1]):m for line in (RAW/'run.log').read_text().splitlines() if (m:=re.search(r'FIT epoch (\d+)/400 step=(\d+) loss=([.\d]+) volume_PSNR=([.\d]+) FULL_span_SSIM=([.\d-]+) best_epoch=(\d+)',line))}
    assert len(logs)==400
    for r in history:
        m=logs[int(r['epoch'])]
        for key,index in [('total',3),('volume_psnr',4),('full_ssim_gt_span',5)]: assert abs(r[key]-float(m[index]))<=.5001e-6
        assert int(r['best_fit_epoch'])==int(m[6])
    maps={};metrics=[];probe=[];central=[];max_recalc=0.
    for tag in stages:
        doc=js(RAW/f'{tag}.json');epoch=doc['metrics']['epoch']
        with np.load(RAW/f'{tag}.npz',allow_pickle=False) as z: maps[tag]={k:z[k].copy() for k in z.files}
        d=maps[tag]; assert all(v.shape==(256,256) and v.dtype==np.float32 and np.isfinite(v).all() for v in d.values())
        for key in [k for k in d if k.startswith('gt_')]: assert np.array_equal(d[key],maps[stages[0]][key])
        if epoch:
            for k,v in doc['metrics'].items():
                if k in history[epoch-1]: assert v==history[epoch-1][k],(epoch,k)
        for kind in ['full','mip']:
            s=scores(d['gt_'+kind],d['pred_'+kind])
            for k in ['mae','psnr_range1','ssim_range1','ssim_gt_span','highpass_correlation','lowpass_correlation']:
                delta=abs(s[k]-doc['metrics'][kind+'_'+k]);max_recalc=max(delta,max_recalc)
                assert delta<1e-6,(tag,kind,k,delta)
            metrics.append({'tag':tag,'epoch':epoch,'projection':kind,**s})
        for row in doc['teacher_probe']:
            probe.append({'tag':tag,'epoch':epoch,**row,
                          'raw_velocity_rmse':math.sqrt(row['raw_velocity_mse']),
                          'endpoint_mae_div_1_minus_t':row['endpoint_mae']/(1-row['t'])})
        for key in ['pred_slice_y','teacher_t0','teacher_t0.5','teacher_t0.9']:
            central.append({'tag':tag,'epoch':epoch,'mode':key,**scores(d['gt_slice_y'],d[key])})
    for r in history:
        expected_total=sum(r[k] for k in ['weighted_velocity','weighted_l1','weighted_contrast','weighted_projection'])
        assert abs(r['total']-expected_total)<1e-6
        best=max(history[:int(r['epoch'])],key=lambda x:x['volume_psnr'])
        assert r['best_fit_epoch']==best['epoch'] and r['best_fit_psnr']==best['volume_psnr']
    meta=js(RAW/'checkpoint_metadata.json')
    for tag in stages[1:]:
        if tag in meta:
            rec=js(RAW/f'{tag}.json')['metrics'];assert rec['model_state_sha256']==meta[tag]['model_state_sha256']
    best=max(history,key=lambda r:r['volume_psnr']);end=history[-1]
    chunks=[]
    for lo,hi in [(1,20),(51,70),(81,100),(181,200),(211,230),(231,250),(251,270),(281,300),(351,370),(381,400)]:
        a=[r for r in history if lo<=r['epoch']<=hi]
        chunks.append({'first_epoch':lo,'last_epoch':hi,**{k:float(np.mean([r[k] for r in a])) for k in a[0] if k not in ['epoch','global_step','case_exposures']}})
    baseline=[]
    prior=ROOT/'analysis/flowoct_trainval_results_20261025/independent_metrics.csv'
    for r in csv.DictReader(prior.open(encoding='utf-8-sig')):
        if r['case_id']=='10308': baseline.append(r)
    prior_maps=Path('D:/data/FSdata/flowoct/projections/train_10308.npz')
    with np.load(prior_maps) as p:
        assert np.array_equal(p['gt_full'],maps['best_fit']['gt_full']) and np.array_equal(p['gt_mip'],maps['best_fit']['gt_mip'])
        multi={k:p['best_at_start_'+k].copy() for k in ['full','mip']}
    last20=history[-20:]
    contrast=float(np.mean([r['contrastloss'] for r in last20]));uniform=math.log(127)
    result={'verification':{'history_epochs':400,'log_rows_match':400,'original_maps_float32_256x256':True,
            'all_stage_gt_arrays_identical':True,'projection_score_max_abs_recalc_error':max_recalc,
            'all_stage_history_values_exact':True,'all_weighted_losses_sum':True,
            'all_best_selection_prefixes_correct':True,'reported_checkpoint_state_hashes_match_stage_reports':True,
            'limitation':'No checkpoint weights or3D prediction volumes in archive; cannot independently rerun network or recalculate full-volume scores'},
            'selected_stage_metrics':[js(RAW/f'{t}.json')['metrics'] for t in stages],
            'projection_metrics':metrics,'teacher_probe':probe,'central_plane_metrics':central,'window_means':chunks,
            'negative_or_zero_full_highpass_epochs':sum(r['full_highpass_correlation']<=0 for r in history),
            'best_by_metric':{k:{'epoch':int(max(history,key=lambda r:r[k])['epoch']),'value':max(r[k] for r in history)} for k in ['volume_psnr','volume_ssim','full_ssim_gt_span','mip_ssim_range1','full_highpass_correlation']},
            'last20_contrast':{'raw_loss_mean':contrast,'uniform_127_candidate_ce':uniform,'difference':contrast-uniform,
                'geometric_mean_true_class_probability':math.exp(-contrast),'uniform_probability':1/127,
                'scalar_loss_share':float(np.mean([r['weighted_contrast'] for r in last20])/np.mean([r['total'] for r in last20]))},
            'time':{'train_minutes':sum(r['train_seconds'] for r in history)/60,'evaluation_minutes':sum(r['eval_seconds'] for r in history)/60,
                   'evaluation_share_of_recorded_train_eval':sum(r['eval_seconds'] for r in history)/sum(r['train_seconds']+r['eval_seconds'] for r in history)},
            'memory':{k:{'min':min(r[k] for r in history),'max':max(r[k] for r in history),'mean':float(np.mean([r[k] for r in history]))} for k in ['train_peak_allocated_gib','eval_peak_allocated_gib']},
            'same_case_prior_multi_case_reference':baseline,
            'change_70_to_best':{'volume_psnr_gain':best['volume_psnr']-history[69]['volume_psnr'],
                   'volume_mae_reduction':1-best['volume_mae']/history[69]['volume_mae'],
                   'full_mae_reduction':1-best['full_mae']/history[69]['full_mae']},
            'best394_to400':{k:end[k]-best[k] for k in ['volume_psnr','volume_ssim','full_ssim_gt_span','mip_ssim_range1','full_highpass_correlation']}}
    savejs(HERE/'analysis.json',result)
    for name,rows in [('independent_projection_metrics',metrics),('teacher_probe_analysis',probe),('central_plane_analysis',central),('window_means',chunks)]:
        with (HERE/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Clean figures include epoch400 explicitly instead of hiding the selector conflict.
    show=['epoch70','epoch100','epoch200','epoch300','best_fit','epoch400']
    fig,axes=plt.subplots(2,7,figsize=(18.2,5.8),constrained_layout=True)
    for i,kind in enumerate(['full','detail']):
        gt=maps['best_fit']['gt_full'];low=float(gt.min());high=float(np.percentile(gt,99))
        arrays=[gt]+[maps[t]['pred_full'] for t in show]
        for j,a in enumerate(arrays):
            if kind=='detail': a=a[40:104,144:208]
            axes[i,j].imshow(a,cmap='gray',vmin=low,vmax=high,interpolation='nearest');axes[i,j].axis('off')
            if i==0: axes[i,j].set_title((['GT']+[str(js(RAW/f'{t}.json')['metrics']['epoch'])+('轮\n体积PSNR最佳' if t=='best_fit' else '轮') for t in show])[j],fontsize=10)
    fig.suptitle('固定病例10308：亮度和大致轮廓在改善，细血管没有被可靠重建\n上：FULL；下：预先固定局部。所有轮次共享GT显示窗口，无增强处理。',fontsize=13)
    fig.savefig(FIG/'01_FULL与固定局部_含400轮.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    xs=[r['epoch'] for r in history]
    for ax,key,title in zip(axes.flat,['volume_psnr','full_ssim_gt_span','full_highpass_correlation','total'],
                            ['体积PSNR：394轮最高','FULL结构SSIM：400轮最高','FULL高频相关：395/400轮不大于0','总损失：下降不代表细血管已经恢复']):
        ax.plot(xs,[r[key] for r in history],lw=1.1,color='#1b6177');ax.axvspan(236,250,color='#bb4a38',alpha=.13);ax.grid(alpha=.18);ax.set_title(title);ax.set_xlabel('更新次数（单病例轮次）')
        if key=='full_highpass_correlation': ax.axhline(0,color='#bb4a38',lw=.8)
    fig.savefig(FIG/'02_训练曲线与异常区间.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
    for key,label,color in [('weighted_velocity','速度项×10','#1b6177'),('weighted_l1','L1×0.5','#cb6f24'),('weighted_contrast','对比项×0.005','#91464d'),('weighted_projection','投影项×0.1','#62764b')]:
        axes[0].plot(xs,[r[key] for r in history],label=label,color=color,lw=1)
    axes[0].set_ylim(0,.08);axes[0].set_title('加权损失：后期对比项约占标量总值49%');axes[0].legend(fontsize=9);axes[0].grid(alpha=.15)
    axes[1].plot(xs,[r['contrastloss'] for r in history],label='原始对比损失',lw=1,color='#91464d')
    axes[1].axhline(uniform,label='均匀127候选基准 ln(127)',ls='--',lw=1,color='#1b6177');axes[1].set_title('对比项贴近均匀基准：不支持结构判别已学成');axes[1].legend(fontsize=9);axes[1].grid(alpha=.15)
    for ax in axes: ax.set_xlabel('更新次数')
    fig.savefig(FIG/'03_损失拆解与对比基准.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12.5,3.8),constrained_layout=True)
    gt=maps['best_fit']['gt_full'];arrays=[gt,maps['best_fit']['pred_full'],maps['epoch400']['pred_full'],multi['full']]
    for ax,a,title in zip(axes,arrays,['GT','单病例394轮 /394步','单病例400轮 /400步','原多病例313轮 /43820步']):
        ax.imshow(a,cmap='gray',vmin=gt.min(),vmax=np.percentile(gt,99),interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('同一个10308、同一GT与显示窗口；更新预算不同，只作参照，不是公平胜负比较',fontsize=11)
    fig.savefig(FIG/'04_同病例历史参照_明确预算差异.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,4,figsize=(12.5,3.8),constrained_layout=True)
    gt=maps['best_fit']['gt_full'];hf=gt-gaussian_filter(gt,1);limit=float(np.percentile(abs(hf),99))
    for ax,a,title in zip(axes,[gt,maps['best_fit']['pred_full'],maps['epoch400']['pred_full'],multi['full']],['GT','单病例394轮','单病例400轮','多病例313轮（参照）']):
        hp=a-gaussian_filter(a,1);ax.imshow(hp,cmap='gray',vmin=-limit,vmax=limit,interpolation='nearest');ax.set_title(title,fontsize=10);ax.axis('off')
    fig.suptitle('仅用于分析的高通分量：噪点不等于血管细节；不能把此图当作增强后的模型结果',fontsize=11)
    fig.savefig(FIG/'05_高通结构对照_仅诊断.png',dpi=170);plt.close(fig)
    print(json.dumps({k:result[k] for k in ['verification','best_by_metric','last20_contrast','time','best394_to400']},ensure_ascii=False,indent=2))


if __name__=='__main__': main()
