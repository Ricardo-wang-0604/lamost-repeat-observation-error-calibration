# -*- coding: utf-8 -*-
"""
10_robustness.py —— 稳健性自查：剔除变星/污染后 χ²_red 是否仍 >> 1

背景
----
`03_repeatability.py` 在全样本上得到 χ²_red(Teff) = 5.875（官方误差若正确应为 1），
第一反应是"官方误差低估"。但**这个结论并不稳健**：

  LAMOST 官方标定误差时明确剔除了变星（DR10 文档 §3.2）。本项目包含全部样本，
  因此变星/双星（参数真的在变）与污染/错配（把不同目标当成同一颗星）
  都会抬高组内离散度，进而抬高 χ²_red。

做法（不依赖外部变星表，自包含）
--------------------------------
1. 逐组算 χ²/dof，按大小排序
2. 依次剔除最离散的 0% / 0.01% / 0.1% / 1% / 5% 的组，重算 χ²_red
3. 若 χ²_red 随少量剔除迅速回落，说明"超出 1"由少数污染组主导，
   而非官方误差公式的系统性偏差

    实测（Teff）：5.875 → 3.370 → 2.143 → 1.233 → 0.853
    即**只剔掉 0.1%（1,252 个组）χ²_red 就掉到 2.143，剔 1% 掉到 1.23**。

4. 再对每个 S/N 分箱做同样修剪（双重控制），排除"污染集中在某个 S/N 段"

产出
----
results/tables/robustness_trim_variables.csv   各参数随剔除比例的 χ²_red
results/tables/robustness_snr_trim.csv         分 S/N 箱 × 剔 1%
results/tables/robustness_report.txt           人读报告

配套：`11_outliers.py` 刻画这些异常组到底是什么天体。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import loader, repeat as rp  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
LOG = []


def say(s=''):
    print(s, flush=True)
    LOG.append(str(s))


def summarize(stats, label):
    r = rp.pooled_repeatability(stats)
    say('  %-18s 组数 %8d  池化σ %10.4f  χ²_red %8.3f  官方err中位 %9.4f  σ/err %6.3f'
        % (label, r['n_groups'], r['pooled_sigma'], r['chi2_red'],
           r['err_median'], r['ratio_sigma_over_err']))
    return r


def main():
    t0 = time.time()
    say('=' * 92)
    say('  稳健性自查：剔除污染/变星后 χ²_red 是否仍 >> 1')
    say('=' * 92)

    df = pd.read_pickle(os.path.join(loader.INTERIM_DIR, 'stellar.pkl'))
    say('载入 %d 行' % len(df))
    df = df[df['gp_id'].notna()]
    df = df[np.isfinite(df['teff'].to_numpy(dtype=np.float64, na_value=np.nan))]

    TRIM = (0.0, 0.0001, 0.001, 0.01, 0.05)
    out_rows = []

    for param in ('teff', 'logg', 'feh'):
        err = rp.ERR_COL[param]
        say('\n' + '-' * 92)
        say('参数 %s   （剔除最离散的尾部组）' % param)
        say('-' * 92)
        st = rp.group_stats(df, key='gp_id', value=param, err=err, min_obs=2)
        if st.empty:
            continue
        st = st[st['chi2'].notna() & (st['dof'] > 0)].copy()
        st['chi2_dof'] = st['chi2'] / st['dof']
        st = st.sort_values('chi2_dof')
        n_all = len(st)

        for q in TRIM:
            sub = st if q == 0 else st.iloc[:int(round(n_all * (1.0 - q)))]
            r = summarize(sub, '剔除尾部 %.2f%%' % (q * 100))
            out_rows.append({'param': param, 'trim_frac': q, **r})

        say('  （最离散的 1%% 组的 χ²/dof 阈值 = %.1f，最大 %.1f）'
            % (st['chi2_dof'].iloc[int(n_all * 0.99)],
               st['chi2_dof'].iloc[-1]))

    R = pd.DataFrame(out_rows)
    R.to_csv(os.path.join(TABLES, 'robustness_trim_variables.csv'),
             index=False, encoding='utf-8-sig')

    # ---------------- 分 S/N 箱内做同样修剪 ----------------
    say('\n' + '=' * 92)
    say('  分 S/N 箱 + 剔除最离散 1% 组（双重控制）')
    say('=' * 92)
    rows2 = []
    for param in ('teff', 'logg', 'feh'):
        err = rp.ERR_COL[param]
        st = rp.group_stats(df, key='gp_id', value=param, err=err, min_obs=2)
        st = st[st['chi2'].notna() & (st['dof'] > 0) & st['snr_median'].notna()].copy()
        st['snr_bin'] = pd.qcut(st['snr_median'], 8, duplicates='drop')
        say('\n  %s:' % param)
        say('  %-18s %10s %12s %12s %12s'
            % ('S/N 箱', '组数', 'χ²_red(全)', 'χ²_red(剔1%)', '结论'))
        for b, sub in st.groupby('snr_bin', observed=True):
            if len(sub) < 100:
                continue
            r0 = rp.pooled_repeatability(sub)
            s2 = sub.sort_values('chi2')
            s2 = s2.iloc[:int(round(len(s2) * 0.99))]
            r1 = rp.pooled_repeatability(s2)
            ok = '仍 >> 1' if r1['chi2_red'] > 1.5 else '已回落到 1 附近'
            say('  %-18s %10d %12.3f %12.3f %12s'
                % ('%.1f-%.1f' % (b.left, b.right), len(sub),
                   r0['chi2_red'], r1['chi2_red'], ok))
            rows2.append({'param': param, 'snr_lo': b.left, 'snr_hi': b.right,
                          'n_groups': len(sub), 'chi2_red_all': r0['chi2_red'],
                          'chi2_red_trim1pct': r1['chi2_red']})
    pd.DataFrame(rows2).to_csv(
        os.path.join(TABLES, 'robustness_snr_trim.csv'), index=False,
        encoding='utf-8-sig')

    with open(os.path.join(TABLES, 'robustness_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    say('-> results/tables/robustness_report.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())
