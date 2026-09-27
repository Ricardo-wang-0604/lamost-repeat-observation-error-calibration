# -*- coding: utf-8 -*-
"""
03_repeatability.py —— 复观一致性与官方误差标定（核心科学结果）

产出
----
results/tables/repeatability_overall.csv     三参数总体标定
results/tables/repeatability_snr_bins.csv    S/N 分箱
results/tables/repeatability_by_plan.csv     按观测计划
results/tables/repeatability_by_subclass.csv 按光谱型
results/tables/repeatability_by_epoch.csv    按年代
results/tables/repeatability_groupsize.csv   按复观次数分箱
results/tables/repeatability_report.txt      人读报告
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

# 项目根按本文件位置向上两级解析（scripts/ → 项目根）。
# 不再硬编码绝对路径 —— 否则别人 clone 到别的目录跑不起来，
# 也无法把项目整体复制到临时目录做安全试跑。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import loader, repeat as rp  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
LOG = []


def say(s=''):
    print(s)
    LOG.append(str(s))


def save(df, name):
    if df is None or len(df) == 0:
        return
    p = os.path.join(TABLES, name)
    df.to_csv(p, encoding='utf-8-sig')
    say('  -> %s (%d 行)' % (name, len(df)))


def main():
    t0 = time.time()
    say('=' * 82)
    say('  LAMOST 复观恒星的参数一致性 与 官方随机误差标定')
    say('=' * 82)

    pkl = os.path.join(loader.INTERIM_DIR, 'stellar.pkl')
    say('载入 %s' % pkl)
    df = pd.read_pickle(pkl)
    say('  %d 行 × %d 列' % df.shape)

    # 仅保留有 gaia 分组键与主参数的观测
    df = df[df['gp_id'].notna()]
    df = df[np.isfinite(df['teff'].to_numpy(dtype=np.float64, na_value=np.nan))]
    say('  过滤后 %d 行' % len(df))

    size = df.groupby('gp_id', observed=True).size()
    n_star = len(size)
    n_rep = int((size >= 2).sum())
    say('  恒星 %d 颗，其中复观星 %d 颗 (%.2f%%)，复观观测 %d 次'
        % (n_star, n_rep, 100.0 * n_rep / n_star, len(df) - n_star))

    # ---------------- 1. 总体标定 ----------------
    say('\n' + '-' * 82)
    say('一、总体误差标定（分组键 = gp_id，要求组内观测 >= 2）')
    say('-' * 82)
    ov, bins = rp.calibration_table(df, key='gp_id', min_obs=2)
    say('  %-6s %10s %12s %12s %10s %10s %10s %10s'
        % ('参数', '组数', 'pooled σ', '官方err中位', 'σ/err', 'chi2_red', '协方差', '判定'))
    for _, r in ov.iterrows():
        say('  %-6s %10d %12.4f %12.4f %10.2f %10.3f %10s %s'
            % (r['value'], r['n_groups'], r['pooled_sigma'], r['err_median'],
               r['ratio_sigma_over_err'], r['chi2_red'], '-',
               '官方误差偏小' if r['chi2_red'] > 1.5 else '自洽'))
    save(ov, 'repeatability_overall.csv')
    save(bins, 'repeatability_snr_bins.csv')

    # ---------------- 2. 按复观次数分箱 ----------------
    say('\n' + '-' * 82)
    say('二、按复观次数分箱（组内观测越多，统计越可靠）')
    say('-' * 82)
    rows = []
    for v in rp.TARGETS:
        st = rp.group_stats(df, key='gp_id', value=v, err=rp.ERR_COL[v], min_obs=2)
        if st.empty:
            continue
        st = st.copy()
        st['nb'] = pd.cut(st['n_obs'], [1, 2, 3, 5, 10, 20, 100],
                          labels=['2', '3', '4-5', '6-10', '11-20', '>20'])
        for b, sub in st.groupby('nb', observed=True):
            if len(sub) < 10:
                continue
            r = rp.pooled_repeatability(sub)
            r['n_obs_bin'] = str(b)
            r['value'] = v
            r['n_stars'] = int(len(sub))
            rows.append(r)
    gs = pd.DataFrame(rows)
    if len(gs):
        say('  %-6s %-7s %9s %12s %12s %10s %10s'
            % ('参数', '次数箱', '恒星数', 'pooled σ', '官方err中位', 'σ/err', 'chi2_red'))
        for _, r in gs.iterrows():
            say('  %-6s %-7s %9d %12.4f %12.4f %10.2f %10.3f'
                % (r['value'], r['n_obs_bin'], r['n_stars'], r['pooled_sigma'],
                   r['err_median'], r['ratio_sigma_over_err'], r['chi2_red']))
    save(gs, 'repeatability_groupsize.csv')

    # ---------------- 3. 按观测计划 ----------------
    say('\n' + '-' * 82)
    say('三、按观测计划（plan_prefix）分列 —— 不同计划的误差行为是否不同')
    say('-' * 82)
    rows = []
    for v in rp.TARGETS:
        t = rp.repeatability_by_subset(df, key='gp_id', value=v, err=rp.ERR_COL[v],
                                       group_col='plan_prefix', min_obs=2)
        if len(t):
            t['value'] = v
            rows.append(t)
    bp = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(bp):
        cols = [c for c in ['plan_prefix', 'value', 'n_groups', 'n_obs', 'pooled_sigma',
                            'err_median', 'ratio_sigma_over_err', 'chi2_red']
                if c in bp.columns]
        say(bp[cols].to_string(index=False))
    save(bp, 'repeatability_by_plan.csv')

    # ---------------- 4. 按光谱型 ----------------
    say('\n' + '-' * 82)
    say('四、按光谱型（subclass）分列')
    say('-' * 82)
    rows = []
    for v in rp.TARGETS:
        t = rp.repeatability_by_subset(df, key='gp_id', value=v, err=rp.ERR_COL[v],
                                       group_col='subclass', min_obs=2)
        if len(t):
            t['value'] = v
            rows.append(t)
    bs = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(bs):
        cols = [c for c in ['subclass', 'value', 'n_groups', 'pooled_sigma', 'err_median',
                            'ratio_sigma_over_err', 'chi2_red'] if c in bs.columns]
        bs2 = bs.sort_values(['value', 'ratio_sigma_over_err'], ascending=[True, False])
        say(bs2[cols].head(30).to_string(index=False))
    save(bs, 'repeatability_by_subclass.csv')

    # ---------------- 5. 按年代 ----------------
    say('\n' + '-' * 82)
    say('五、按观测年代（复观散布是否随观测季变化）')
    say('-' * 82)
    df = df.copy()
    if 'mjd' in df.columns:
        # MJD 0 = 1858-11-17，故 yyyy = 1858.879 + mjd/365.25。
        # 原先写成 1858.0 会让标签整体偏早 0.879 年，并把 MJD<55883.2
        # （即 2011-11-18 之前）的 40,816 行判为 NaN 后静默丢弃。
        yr = 1858.879 + df['mjd'].to_numpy(dtype=np.float64) / 365.25
        df['epoch'] = pd.cut(yr, [-np.inf, 2013.879, 2015.879, 2017.879, 2019.879, np.inf],
                             labels=['2011-12', '2013-14', '2015-16', '2017-18', '2019-22'])
        n_drop = int(df['epoch'].isna().sum())
        say('  MJD→年 使用常数 1858.879；日期范围 %.1f – %.1f'
            % (yr.min(), yr.max()))
        if n_drop:
            say('  ⚠ 有 %d 行落在分箱之外' % n_drop)
        rows = []
        for v in rp.TARGETS:
            t = rp.repeatability_by_subset(df, key='gp_id', value=v, err=rp.ERR_COL[v],
                                           group_col='epoch', min_obs=2)
            if len(t):
                t['value'] = v
                rows.append(t)
        be = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        if len(be):
            cols = [c for c in ['epoch', 'value', 'n_groups', 'pooled_sigma', 'err_median',
                                'ratio_sigma_over_err', 'chi2_red'] if c in be.columns]
            say(be[cols].to_string(index=False))
        save(be, 'repeatability_by_epoch.csv')

    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    with open(os.path.join(TABLES, 'repeatability_report.txt'), 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
