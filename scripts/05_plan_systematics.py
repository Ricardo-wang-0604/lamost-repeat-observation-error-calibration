# -*- coding: utf-8 -*-
"""
05_plan_systematics.py —— 跨观测计划的系统偏差（配对比较）

动机
----
LAMOST 的部分恒星被**不同观测计划**重复观测。例如一颗 Hipparcos 标准星
在其 36 次观测里横跨了 HIP / TD / KP / KII / HD 五个 planid：

    J061628.33+232933.1:  HIP29425K201(26) + TD061156N231225K01(4)
                          + HIP29425K2(2) + KP/KII/HD 各 1

同一颗恒星的真实参数是常数，因此**同一颗星在两个计划下的参数差**
就是计划间的系统偏差（叠加观测噪声）。方法：

  1. 对每颗星、每个 plan_prefix，算加权均值  xbar_{g,p}
  2. 对计划的每一对 (p, q)，取同时存在于两者的星，做**配对差** d = xbar_{g,p} - xbar_{g,q}
  3. 统计 d 的均值（系统偏移）与标准差（随机散布）
  4. 与 σ_pooled/√N 比较，判断偏移是否显著

为什么配对很重要
----------------
直接比较两组星的平均值会被**不同恒星群体的参数分布差异**污染
（TD 场和 HD 巡天选的是不同天体）。配对比较自动消掉恒星本身的参数，
只剩下计划效应 —— 这与风电项目里"同一场站不同模式比较"是同一思路。

产出
----
results/tables/plan_pair_counts.csv    各计划对的公共星数
results/tables/plan_pair_offsets.csv   配对偏移与显著性
results/tables/plan_means.csv          各计划的参数均值（对照）
results/tables/plan_systematics_report.txt
"""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
sys.path.insert(0, ROOT)
from src import loader  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
LOG = []
TARGETS = ['teff', 'logg', 'feh']


def say(s=''):
    print(s)
    LOG.append(str(s))


def per_star_per_plan(df, target, min_snr=10.0):
    """去低信噪比后，算 (gp_id, plan_prefix) 的加权均值与计数。"""
    use = df[['gp_id', 'plan_prefix', target]].copy()
    use = use[use['plan_prefix'].notna()]
    if 'snrr' in df.columns:
        use['snrr'] = df['snrr'].to_numpy()
        use = use[use['snrr'].notna() & (use['snrr'] >= min_snr)]
    use = use[use[target].notna()]
    use['gp_id'] = use['gp_id'].astype('int64')
    g = use.groupby(['gp_id', 'plan_prefix'], observed=True)[target]
    out = g.agg(['mean', 'size']).reset_index()
    out.columns = ['gp_id', 'plan_prefix', 'mean', 'n']
    return out


def pair_offsets(pp, min_common=30):
    """对所有计划对做配对比较。"""
    plans = sorted(pp['plan_prefix'].unique())
    wide = pp.pivot_table(index='gp_id', columns='plan_prefix', values='mean',
                          observed=True)
    cnt = pp.pivot_table(index='gp_id', columns='plan_prefix', values='n',
                         observed=True)
    rows = []
    for a, b in itertools.combinations(plans, 2):
        if a not in wide.columns or b not in wide.columns:
            continue
        d = (wide[a] - wide[b]).dropna()
        if len(d) < min_common:
            continue
        # 配对差的均值的标准误 → 显著性
        sem = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan
        rows.append({
            'plan_a': a, 'plan_b': b,
            'n_common': int(len(d)),
            'mean_diff': float(d.mean()),
            'median_diff': float(d.median()),
            'std_diff': float(d.std(ddof=1)) if len(d) > 1 else np.nan,
            'sem': sem,
            't_stat': float(d.mean() / sem) if sem and sem > 0 else np.nan,
        })
    return pd.DataFrame(rows), wide, cnt


def main():
    t0 = time.time()
    say('=' * 84)
    say('  跨观测计划的系统偏差（配对比较）')
    say('=' * 84)

    df = pd.read_pickle(os.path.join(loader.INTERIM_DIR, 'stellar.pkl'))
    say('载入 %d 行' % len(df))
    df['gp_id'] = pd.to_numeric(df['gp_id'], errors='coerce')
    df = df[df['gp_id'].notna()].copy()
    df['gp_id'] = df['gp_id'].astype('int64')

    if 'plan_prefix' not in df.columns:
        df['plan_prefix'] = (df['planid'].astype('string')
                             .str.extract(r'^([A-Za-z]+)', expand=False))
    df['plan_prefix'] = df['plan_prefix'].astype('string').fillna('UNK')

    say('\n各计划观测数:')
    vc = df['plan_prefix'].value_counts()
    for k, v in vc.items():
        say('  %-8s %10d' % (k, v))

    all_pairs = []
    all_counts = []
    for tgt in TARGETS:
        say('\n' + '-' * 84)
        say('参数 %s' % tgt)
        say('-' * 84)
        pp = per_star_per_plan(df, tgt)
        say('  (恒星 × 计划) 组合数: %d' % len(pp))
        multi = pp.groupby('gp_id', observed=True).size()
        say('  被 >=2 个计划观测过的恒星: %d 颗' % int((multi >= 2).sum()))

        po, wide, cnt = pair_offsets(pp, min_common=30)
        if po.empty:
            say('  !! 没有任何计划对有 >=30 颗公共星')
            continue
        po['target'] = tgt
        po = po.reindex(po['n_common'].sort_values(ascending=False).index)
        say('\n  %-8s %-8s %9s %12s %12s %10s %10s'
            % ('计划A', '计划B', '公共星数', '平均差(A-B)', '差的std', 'SEM', 't'))
        for _, r in po.head(25).iterrows():
            say('  %-8s %-8s %9d %12.4f %12.4f %10.4f %10.2f'
                % (r['plan_a'], r['plan_b'], r['n_common'], r['mean_diff'],
                   r['std_diff'], r['sem'], r['t_stat']))
        all_pairs.append(po)

        # 各计划的森林图式均值（仅用多计划星，保证可比）
        multi_ids = multi[multi >= 2].index
        sub = pp[pp['gp_id'].isin(multi_ids)]
        if len(sub):
            m = sub.groupby('plan_prefix', observed=True).agg(
                n_star=('mean', 'size'), mean=('mean', 'mean'), std=('mean', 'std')
            ).reset_index()
            m['target'] = tgt
            all_counts.append(m)
            say('\n  仅在"多计划星"子集内，各计划的参数均值（可比）:')
            for _, r in m.iterrows():
                say('    %-8s 星数 %6d  均值 %10.4f  散布 %9.4f'
                    % (r['plan_prefix'], r['n_star'], r['mean'], r['std']))

    PAIRS = pd.concat(all_pairs, ignore_index=True) if all_pairs else pd.DataFrame()
    MEANS = pd.concat(all_counts, ignore_index=True) if all_counts else pd.DataFrame()
    if len(PAIRS):
        PAIRS.to_csv(os.path.join(TABLES, 'plan_pair_offsets.csv'),
                     index=False, encoding='utf-8-sig')
    if len(MEANS):
        MEANS.to_csv(os.path.join(TABLES, 'plan_means.csv'),
                     index=False, encoding='utf-8-sig')

    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    with open(os.path.join(TABLES, 'plan_systematics_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
