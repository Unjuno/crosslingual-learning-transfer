#!/usr/bin/env python3
"""Independently check R2b raw curves, summaries and measured execution traces.

No training libraries required. An input/trace error yields UNCERTAIN, never an
experimental failure. The science gates are the original locked R2b gates.
B_step/span are optimizer-update counts; one update contains 2048 byte targets.
EN_nll/JA_post_nll are natural-log cross-entropies per byte target.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

SEEDS = tuple(range(32000, 32010))
ARMS = ('JA_nat', 'JA_TJA', 'JA_TEN', 'JA_TRU')
ALL_ARMS = ('REF_EN',) + ARMS
GRID = tuple(range(0, 601, 10))
PROTOCOL = 'protocols/natural_ja_en_s2_halfweight_tradeoff_2026-09-06.json'
CORE_BLOB = 'aab161824538e29f71f2a5f00fd44364f2de9bd4'
WRAPPER_BLOB = '9dcc9f5a4c4c76ed62e173ffb99b9f911de84996'
CORPUS = {
 'en_train':'74f59db34b5b8334cf463cf0bcf5f355e54cc7de98cc4eb35ef379ecb42e8258',
 'en_eval':'15221408b2c4c2fffee8a5f16feb651f2de5947e85906867f119fdf7210797da',
 'ja_train':'5a0d86cebeb5e9f71c86ea1bcf619c8a42bb21d04ed2d37ce35f5aee58bffd58',
 'ja_eval':'5a44582a08d6e3c48c610d663b22642c8b8612dc19f25a326b2ce906a504e7e4',
 'ru_train':'12a53e456480b56977c548021231c9adf9b4cee97902f6d1acb67e6cbb0d46ed',
 'ru_eval':'52a1cd50a20645577e988f6023dcf1853e856f5babd180a0e1d620029c689800',
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def number(value, label: str) -> float:
    v = float(value)
    require(math.isfinite(v), f'{label}: nonfinite')
    return v


def close(a: float, b: float, label: str) -> None:
    require(math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9), f'{label}: {a} != {b}')


def canonical_bool(value: str) -> bool:
    require(value in ('True', 'False'), 'noncanonical bool')
    return value == 'True'


def read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def best_curve(points: list[tuple[int,float]]) -> list[tuple[int,float]]:
    require(bool(points), 'empty curve')
    out=[]; previous=-1; best=math.inf
    for step, loss in sorted(points):
        require(step > previous, 'duplicate or negative curve step')
        loss=number(loss, 'NLL'); require(loss >= 0, 'negative NLL')
        best=min(best,loss);out.append((step,best)); previous=step
    return out


def crossing(points, threshold: float) -> float | None:
    c=best_curve(points)
    if c[0][1] <= threshold:
        return float(c[0][0])
    for (x0,y0),(x1,y1) in zip(c,c[1:]):
        if y1 <= threshold:
            require(y0 > y1, 'invalid crossing segment')
            return x0+(x1-x0)*(y0-threshold)/(y0-y1)
    return None


def span(points, high: float, low: float) -> float | None:
    require(math.isfinite(high) and math.isfinite(low) and high > low,
            'reference performance interval must have positive width')
    hi,lo=crossing(points,high),crossing(points,low)
    return None if hi is None or lo is None else lo-hi


def sign_p(k: int, n: int=10) -> float:
    require(0 <= k <= n and n > 0, 'invalid sign count')
    return sum(math.comb(n,j) for j in range(k,n+1))/(2**n)


def describe(values: list[float]) -> dict:
    require(bool(values), 'empty descriptive sample')
    return {'n':len(values),'mean':statistics.mean(values),
            'median':statistics.median(values),'min':min(values),'max':max(values),
            'sample_sd':statistics.stdev(values) if len(values)>1 else None,
            'standard_error_across_seeds':statistics.stdev(values)/math.sqrt(len(values)) if len(values)>1 else None}


def expected_sources() -> dict:
    counts={'base/base':{'ja_train':400},
            'teachers/T_JA':{'ja_train':400},
            'teachers/T_EN':{'ja_train':200,'en_train':200},
            'teachers/T_RU':{'ja_train':200,'ru_train':200}}
    for arm in ARMS:counts['phase1/'+arm]={'ja_train':200}
    for arm in ALL_ARMS:counts['phase2_'+arm+'/phase2_'+arm]={'en_train':600}
    return counts


def verify_seed(root: Path, seed: int) -> dict:
    paths={kind:root/f'natural_s2_seed{seed}_{kind}' for kind in
           ('curves.csv','summary.csv','audit.json')}
    paths['trace']=root/f'seed{seed}_execution_trace.json'
    for p in paths.values():require(p.is_file(),f'missing {p.name}')
    summary=read_csv(paths['summary.csv']); curve_rows=read_csv(paths['curves.csv'])
    require(len(summary)==4, 'four summary conditions required')
    require(len(curve_rows)==305, 'five complete61-point curves required')
    by={r['condition']:r for r in summary}
    require(set(by)==set(ARMS), 'summary conditions mismatch/duplication')
    curves={a:[] for a in ALL_ARMS}
    for r in summary+curve_rows:
        require(int(r['seed'])==seed,'seed label mismatch')
        require(r.get('smoke_only')=='False' and r.get('validation_only')=='False',
                'non-scientific or missing evidence flags')
    for r in curve_rows:
        require(r['condition'] in curves,'unexpected curve condition')
        curves[r['condition']].append((int(r['B_step']),number(r['EN_nll'],'curve NLL')))
    for arm,points in curves.items():
        curves[arm]=sorted(points)
        require(tuple(x for x,_ in curves[arm])==GRID,f'{arm}: missing/duplicate grid points')
        best_curve(curves[arm])
    ref=best_curve(curves['REF_EN']);top=ref[0][1];bottom=ref[-1][1]
    require(top>bottom,'reference failed to improve')
    high=bottom+0.8*(top-bottom); low=bottom+0.2*(top-bottom)
    spans={a:span(curves[a],high,low) for a in ARMS}
    for arm,row in by.items():
        close(number(row['reference_high'],'high'),high,'reference_high')
        close(number(row['reference_low'],'low'),low,'reference_low')
        close(number(row['EN_step0_nll'],'step0'),curves[arm][0][1],'step0')
        require(canonical_bool(row['censored'])==(spans[arm] is None),'censoring flag mismatch')
        if spans[arm] is None:require(row['span_updates'] in ('','nan'),'censored span must be absent')
        else:close(number(row['span_updates'],'span'),spans[arm],'recomputed span')
        require(number(row['JA_post_nll'],'Japanese NLL')>0,'nonpositive Japanese NLL')
    audit=json.loads(paths['audit.json'].read_text());trace=json.loads(paths['trace'].read_text())
    require(audit['seed']==seed and trace['seed']==seed,'JSON seed mismatch')
    require(audit['scientific_evidence'] is True and audit['smoke_only'] is False and audit['validation_only'] is False,'invalid evidence flags')
    require(audit['corpus_sha256']==CORPUS and audit['corpus_hashes_match_frozen_manifest'] is True,'corpus hash mismatch')
    require(audit['protocol']==PROTOCOL and audit['hidden_distillation_weight']==0.5 and audit['tradeoff_followup'] is True and audit['parent_locked_verdict']=='FAIL','protocol/weight mismatch')
    require(audit['prior_result_seed_hits']==[],'reused seed')
    require(audit['student_phase0_sources']=={'ja':400,'en':0,'ru':0},'wrong base sources')
    require(audit['student_phase1_sources']=={'ja':200,'en':0,'ru':0},'wrong phase1 sources')
    require(audit['teacher_source_counts']=={'T_JA':{'ja':400,'en':0,'ru':0},'T_EN':{'ja':200,'en':200,'ru':0},'T_RU':{'ja':200,'en':0,'ru':200}},'wrong teacher counts')
    require(audit['schedule']=={'base_steps':400,'teacher_steps':400,'phase1_steps':200,'phase2_steps':600,'eval_every':10},'changed schedule')
    require(audit['fixed_byte_vocab']==256 and audit['teacher_target_slot_positions_identical'] is True,'tokenizer/schedule audit mismatch')
    require(trace['complete'] is True and trace['validation_only'] is False,'incomplete trace')
    require(trace['core_blob']==CORE_BLOB and trace['halfweight_wrapper_blob']==WRAPPER_BLOB,'different executed code')
    require(trace['hidden_distillation_weight']==0.5 and trace['steps_observed']==5400,'wrong executed weight/update budget')
    require(trace['actual_source_counts']==expected_sources(),'measured sources mismatch')
    require(trace['model_update_counts']=={k:sum(v.values()) for k,v in expected_sources().items()},'measured update counts mismatch')
    for name in ('phase1_same_batches','phase2_same_batches','pre_phase2_student_only_ja','teacher_target_slots_match','deterministic'):
        require(trace[name] is True,f'measured audit failed: {name}')
    digests=trace['actual_batch_sha256']
    for phase,n in [('phase1/',4),('phase2_',5)]:
        vals=[v for k,v in digests.items() if k.startswith(phase)]
        require(len(vals)==n and len(set(vals))==1 and len(vals[0])==64,'measured common batches mismatch')
    require(trace['torch_num_threads']==1,'unexpected torch thread count')
    ten,tja=spans['JA_TEN'],spans['JA_TJA']
    censored=ten is None or tja is None
    return {'seed':seed,'spans':spans,'primary_effect':None if censored else ten-tja,
            'saving_vs_TJA':None if censored or tja<=0 else 1-ten/tja,
            'JA_TEN_nll':float(by['JA_TEN']['JA_post_nll']),
            'JA_TJA_nll':float(by['JA_TJA']['JA_post_nll']),
            'JA_nat_nll':float(by['JA_nat']['JA_post_nll']),
            'EN_step0':{a:curves[a][0][1] for a in ARMS},
            'EN_step600':{a:curves[a][-1][1] for a in ARMS},
            'practical_effect':None if ten is None or spans['JA_nat'] is None else ten-spans['JA_nat'],
            'specificity_effect':None if ten is None or spans['JA_TRU'] is None else ten-spans['JA_TRU'],
            'censored':censored,'all_audits_pass':True,
            'runtime_seconds':trace['runtime_seconds'],
            'input_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}}


def verify(root: Path) -> dict:
    rows=[verify_seed(root,s) for s in SEEDS]
    favorable=sum(r['primary_effect'] is not None and r['primary_effect']<0 for r in rows)
    savings=[r['saving_vs_TJA'] for r in rows]
    mean_saving=statistics.mean(savings) if all(v is not None for v in savings) else None
    ja_ten=statistics.mean(r['JA_TEN_nll'] for r in rows)
    ja_tja=statistics.mean(r['JA_TJA_nll'] for r in rows)
    gates={'direction':favorable>=9,'sign_p':sign_p(favorable)<=0.05,
           'saving':mean_saving is not None and mean_saving>=0.05,
           'Japanese_retention':ja_ten<=1.01*ja_tja,
           'no_censoring':not any(r['censored'] for r in rows),'audits':True}
    return {'verdict':'PASS' if all(gates.values()) else 'FAIL',
            'verification_level':'raw_curve_recomputation_and_measured_execution_trace',
            'gates':gates,'n_seeds':10,'n_summary_rows':40,'n_curve_rows':3050,
            'n_optimizer_updates':54000,'favorable_primary_signs':favorable,
            'exact_one_sided_sign_p':sign_p(favorable),
            'mean_saving_vs_TJA':mean_saving,
            'Japanese_relative_NLL_degradation':ja_ten/ja_tja-1,
            'mean_JA_TEN_nll':ja_ten,'mean_JA_TJA_nll':ja_tja,
            'savings_description':describe(savings) if mean_saving is not None else None,
            'runtime_description_seconds':describe([r['runtime_seconds'] for r in rows]),
            'seed_results':rows}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('input_dir',type=Path)
    ap.add_argument('--json-out',type=Path,required=True);args=ap.parse_args()
    try:
        report=verify(args.input_dir);code=0
    except (ValueError,KeyError,OSError,TypeError,ZeroDivisionError) as e:
        report={'verdict':'UNCERTAIN','reason':'input or execution-trace validation failed','error':str(e)};code=2
    args.json_out.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n'
    args.json_out.write_text(text);print(text)
    raise SystemExit(code)


if __name__=='__main__':main()
