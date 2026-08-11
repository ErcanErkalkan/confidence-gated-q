from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

ALGOS=["SAC","CrossQ","TQC"]
ENVS=["HalfCheetah-v5","Walker2d-v5"]
SEEDS=list(range(22000,22005))
CONDS=["NOMINAL","OBS_DELAY_1","ACT_GAIN_075","COMBINED_DELAY_GAIN"]
PRIMARY_SHIFTS=["OBS_DELAY_1","ACT_GAIN_075"]


def holm(pvals):
    p=np.asarray(pvals,dtype=float)
    m=len(p); order=np.argsort(p); adj=np.empty(m,float); running=0.0
    for rank,idx in enumerate(order):
        val=(m-rank)*p[idx]
        running=max(running,val)
        adj[idx]=min(1.0,running)
    return adj


def safe_wilcoxon(x):
    x=np.asarray(x,dtype=float)
    if np.allclose(x,0):
        return 0.0,1.0
    try:
        r=stats.wilcoxon(x,alternative="two-sided",zero_method="wilcox")
        return float(r.statistic),float(r.pvalue)
    except Exception:
        return float("nan"),float("nan")


def paired_stats(diff, idx):
    diff=np.asarray(diff,dtype=float)
    t=stats.ttest_1samp(diff,0.0,nan_policy="raise")
    wstat,wp=safe_wilcoxon(diff)
    rng=np.random.default_rng(56001+idx)
    boots=np.empty(10000,float)
    n=len(diff)
    for b in range(len(boots)):
        boots[b]=np.mean(diff[rng.integers(0,n,size=n)])
    lo,hi=np.quantile(boots,[0.025,0.975])
    return {
        "n_pairs":n,"mean_paired_difference":float(np.mean(diff)),
        "median_paired_difference":float(np.median(diff)),
        "paired_bootstrap_ci95_low":float(lo),"paired_bootstrap_ci95_high":float(hi),
        "paired_wins":int(np.sum(diff>0)),"paired_losses":int(np.sum(diff<0)),"paired_ties":int(np.sum(diff==0)),
        "paired_t_stat":float(t.statistic),"paired_t_p":float(t.pvalue),
        "wilcoxon_stat":wstat,"wilcoxon_p":wp,
    }


def cvar10(values):
    a=np.sort(np.asarray(values,dtype=float)); k=max(1,int(math.ceil(0.10*len(a))))
    return float(np.mean(a[:k]))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results-root',type=Path,default=Path('results/continuous_control'))
    ap.add_argument('--out',type=Path,default=Path('results/continuous_control/analysis'))
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    root=args.results_root/'supplemental'
    summaries=[]; episode_parts=[]; checkpoint_parts=[]; errors=[]
    for algo in ALGOS:
      for env in ENVS:
       for seed in SEEDS:
        d=root/algo/env/f'seed_{seed}'
        sf=d/'run_summary.json'; ef=d/'episode_results.csv'; cf=d/'checkpoint_results.csv'
        if not sf.exists(): errors.append(f'missing run_summary: {d}'); continue
        s=json.loads(sf.read_text(encoding='utf-8')); summaries.append(s)
        if s.get('status')!='COMPLETE': errors.append(f'non-COMPLETE summary: {d}')
        if not ef.exists(): errors.append(f'missing episode_results: {d}'); continue
        e=pd.read_csv(ef); e['_source']=str(d); episode_parts.append(e)
        if len(e)!=120: errors.append(f'expected 120 final rows, got {len(e)}: {d}')
        if not cf.exists(): errors.append(f'missing checkpoint_results: {d}'); continue
        c=pd.read_csv(cf); c['algorithm']=algo; c['environment']=env; c['training_seed']=seed; checkpoint_parts.append(c)
        if len(c)!=20: errors.append(f'expected 20 checkpoint rows, got {len(c)}: {d}')
    if len(summaries)!=30: errors.append(f'expected 30 summaries, got {len(summaries)}')
    if errors:
        audit={'status':'FAIL','errors':errors}
        (args.out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        print(json.dumps(audit,indent=2)); return 2
    ep=pd.concat(episode_parts,ignore_index=True)
    ck=pd.concat(checkpoint_parts,ignore_index=True)
    # strict registered-grid checks
    if set(ep['algorithm'])!=set(ALGOS) or set(ep['environment'])!=set(ENVS) or set(ep['condition'])!=set(CONDS):
        raise RuntimeError('Final result grid differs from locked grid')
    # trained-seed metrics
    rows=[]
    for (algo,env,seed,cond),g in ep.groupby(['algorithm','environment','training_seed','condition'],sort=True):
        dec=float(g['support_decisions'].sum()); sup=float(g['supported_decisions'].sum())
        unhealthy=None if env!='Walker2d-v5' else float(pd.to_numeric(g['unhealthy_termination'],errors='coerce').fillna(False).astype(bool).mean())
        rows.append({
            'algorithm':algo,'environment':env,'training_seed':int(seed),'condition':cond,
            'mean_return':float(g['episode_return'].mean()),'median_return':float(g['episode_return'].median()),
            'cvar10_return':cvar10(g['episode_return'].to_numpy()),
            'decision_weighted_support_coverage':sup/dec if dec else float('nan'),
            'episode_mean_support_coverage':float(g['support_coverage'].mean()),
            'exact_recurrence_rate_episode_mean':float(g['exact_recurrence_rate'].mean()),
            'unhealthy_termination_rate':unhealthy,'truncation_rate':float(g['truncated'].astype(bool).mean()),
            'mean_critic_gap':float(pd.to_numeric(g['critic_gap_mean'],errors='coerce').mean()),
            'mean_tqc_quantile_iqr':float(pd.to_numeric(g['tqc_quantile_iqr_mean'],errors='coerce').mean()) if algo=='TQC' else None,
            'final_eval_episodes':int(len(g)),'final_eval_decisions':int(dec),
        })
    sm=pd.DataFrame(rows); sm.to_csv(args.out/'seed_metrics.csv',index=False)
    # checkpoints and AUC
    cks=(ck.groupby(['algorithm','environment','training_seed','timestep'],as_index=False)
           .agg(mean_checkpoint_return=('episode_return','mean'),median_checkpoint_return=('episode_return','median')))
    cks.to_csv(args.out/'checkpoint_seed_metrics.csv',index=False)
    aucrows=[]
    for (algo,env,seed),g in cks.groupby(['algorithm','environment','training_seed']):
        g=g.sort_values('timestep'); x=g['timestep'].to_numpy(float); y=g['mean_checkpoint_return'].to_numpy(float)
        if len(g)!=2 or int(x[0])!=50000 or int(x[-1])!=100000:
            raise RuntimeError(f'Checkpoint schedule incomplete for {algo}/{env}/{seed}')
        area=float(np.trapezoid(y,x) if hasattr(np, 'trapezoid') else np.trapz(y,x)); aucrows.append({'algorithm':algo,'environment':env,'training_seed':int(seed),'return_auc':area,'normalized_return_auc':area/(x[-1]-x[0])})
    pd.DataFrame(aucrows).to_csv(args.out/'return_auc.csv',index=False)
    # S1
    s1=[]; idx=0
    for shift in PRIMARY_SHIFTS:
      for env in ENVS:
       for challenger in ['CrossQ','TQC']:
        a=sm[(sm.algorithm==challenger)&(sm.environment==env)&(sm.condition==shift)].sort_values('training_seed')
        b=sm[(sm.algorithm=='SAC')&(sm.environment==env)&(sm.condition==shift)].sort_values('training_seed')
        if a.training_seed.tolist()!=SEEDS or b.training_seed.tolist()!=SEEDS: raise RuntimeError('S1 seed mismatch')
        st=paired_stats(a.mean_return.to_numpy()-b.mean_return.to_numpy(),idx); idx+=1
        s1.append({'family':'S1','challenger':challenger,'reference':'SAC','environment':env,'condition':shift,**st})
    s1df=pd.DataFrame(s1); s1df['holm_p']=holm(s1df['paired_t_p']); s1df['holm_significant_0_05']=s1df.holm_p<0.05
    s1df.insert(0,'inferential_status','SUPPLEMENTAL_NONCONFIRMATORY_SENSITIVITY'); s1df.to_csv(args.out/'S1_supplemental_controller_contrasts.csv',index=False)
    # S2
    s2=[]; idx=0
    for algo in ALGOS:
      for env in ENVS:
       nom=sm[(sm.algorithm==algo)&(sm.environment==env)&(sm.condition=='NOMINAL')].sort_values('training_seed')
       for shift in PRIMARY_SHIFTS:
        sh=sm[(sm.algorithm==algo)&(sm.environment==env)&(sm.condition==shift)].sort_values('training_seed')
        if nom.training_seed.tolist()!=SEEDS or sh.training_seed.tolist()!=SEEDS: raise RuntimeError('S2 seed mismatch')
        diff=sh.decision_weighted_support_coverage.to_numpy()-nom.decision_weighted_support_coverage.to_numpy()
        st=paired_stats(diff,100+idx); idx+=1
        s2.append({'family':'S2','algorithm':algo,'environment':env,'condition':shift,'contrast':'shift_minus_nominal_support_coverage',**st})
    s2df=pd.DataFrame(s2); s2df['holm_p']=holm(s2df['paired_t_p']); s2df['holm_significant_0_05']=s2df.holm_p<0.05
    s2df.insert(0,'inferential_status','SUPPLEMENTAL_NONCONFIRMATORY_SENSITIVITY'); s2df.to_csv(args.out/'S2_supplemental_support_contrasts.csv',index=False)
    # descriptive condition summary
    desc=(sm.groupby(['algorithm','environment','condition'],as_index=False)
            .agg(mean_of_seed_mean_return=('mean_return','mean'),sd_seed_mean_return=('mean_return','std'),
                 mean_support_coverage=('decision_weighted_support_coverage','mean'),sd_support_coverage=('decision_weighted_support_coverage','std'),
                 mean_cvar10=('cvar10_return','mean')))
    desc.to_csv(args.out/'condition_summary.csv',index=False)
    audit={'status':'PASS','inferential_status':'SUPPLEMENTAL_NONCONFIRMATORY','complete_runs':30,'final_episode_rows':int(len(ep)),'checkpoint_episode_rows':int(len(ck)),
           'S1_tests':int(len(s1df)),'S2_tests':int(len(s2df)),'analysis_bootstrap_base_seed':56001}
    (args.out/'audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(audit,indent=2)); return 0

if __name__=='__main__': raise SystemExit(main())
