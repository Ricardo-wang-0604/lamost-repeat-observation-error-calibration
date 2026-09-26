# -*- coding: utf-8 -*-
"""
07_figures.py —— 出图

产出 results/figures/ 下的 PNG：
  fig1_repeat_dist.png       复观组规模分布
  fig2_sigma_vs_err.png      实测池化 σ 对 官方误差（三参数）
  fig3_conv_snr.png          复观次数收敛：σ 随组内观测数下降 → 逼近真值
  fig4_by_plan.png           按观测计划的 σ/err 与 chi2_red
  fig5_leakage.png           标签泄漏：随机划分 vs 分组划分
  fig6_snr_bins.png          σ/err 随信噪比变化
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
sys.path.insert(0, ROOT)
from src import loader  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
FIGS = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIGS, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _save(fig, name):
    p = os.path.join(FIGS, name)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print('  -> %s' % p)


def fig1():
    df = pd.read_pickle(os.path.join(loader.INTERIM_DIR, 'stellar.pkl'))
    size = df.groupby('gp_id', observed=True).size()
    del df
    dist = size.value_counts().sort_index()
    dist = dist[dist.index <= 15]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(dist.index.astype(int), dist.values, color='#2b6cb0')
    ax.set_yscale('log')
    ax.set_xlabel('Repeated observations per star (gp_id)')
    ax.set_ylabel('Number of stars (log)')
    ax.set_title('LAMOST DR10: repeat-observation multiplicity')
    for x, y in zip(dist.index.astype(int), dist.values):
        ax.text(x, y * 1.15, str(int(y)), ha='center', fontsize=7)
    _save(fig, 'fig1_repeat_dist.png')


def fig2():
    p = os.path.join(TABLES, 'repeatability_overall.csv')
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    lim = max(df['pooled_sigma'].max(), df['err_median'].max()) * 1.25
    ax.plot([0, lim], [0, lim], 'k--', lw=1, label='quoted error = observed scatter')
    ax.scatter(df['err_median'], df['pooled_sigma'], s=90, zorder=5,
               color='#c53030')
    for _, r in df.iterrows():
        ax.annotate(r['value'], (r['err_median'], r['pooled_sigma']),
                    textcoords='offset points', xytext=(8, 4), fontsize=10)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel('Median quoted error (same units as sigma)')
    ax.set_ylabel('Pooled repeat scatter sigma')
    ax.set_title('Observed repeatability vs. quoted error')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, 'fig2_sigma_vs_err.png')


def fig3():
    p = os.path.join(TABLES, 'repeatability_groupsize.csv')
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, param in zip(axes, ['teff', 'logg', 'feh']):
        s = df[df['value'] == param]
        if s.empty:
            continue
        x = np.arange(len(s))
        ax.errorbar(x, s['pooled_sigma'], fmt='o-', color='#2b6cb0',
                    label='pooled sigma')
        ax.errorbar(x, s['err_median'], fmt='s--', color='#c53030',
                    label='median quoted err')
        ax2 = ax.twinx()
        ax2.plot(x, s['chi2_red'], '^:', color='#276749', alpha=0.8,
                 label='chi2_red')
        ax2.axhline(1.0, color='gray', lw=0.8, ls=':')
        ax2.set_ylabel('chi2_red', color='#276749')
        ax.set_xticks(x)
        ax.set_xticklabels(s['n_obs_bin'].astype(str))
        ax.set_xlabel('Observations per star')
        ax.set_ylabel('sigma / err')
        ax.set_title(param)
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='best')
    fig.suptitle('Convergence with the number of repeat observations', fontsize=11)
    _save(fig, 'fig3_conv_snr.png')


def fig4():
    p = os.path.join(TABLES, 'repeatability_by_plan.csv')
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    s = df[df['value'] == 'teff'].dropna(subset=['ratio_sigma_over_err'])
    if s.empty:
        return
    s = s.sort_values('n_groups', ascending=False).head(14)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(s))
    ax.bar(x, s['ratio_sigma_over_err'], color='#b7791f', label='sigma / err')
    ax.axhline(1.0, color='k', lw=1, ls='--', label='self-consistent')
    ax2 = ax.twinx()
    ax2.plot(x, s['chi2_red'], 'o-', color='#c53030', label='chi2_red')
    ax2.axhline(1.0, color='#c53030', lw=0.8, ls=':')
    ax2.set_ylabel('chi2_red', color='#c53030')
    ax.set_xticks(x)
    ax.set_xticklabels(['%s\n(%d)' % (a, int(b)) for a, b in
                        zip(s['plan_prefix'], s['n_groups'])], fontsize=8)
    ax.set_xlabel('Observing plan (number of repeat groups)')
    ax.set_ylabel('sigma / err  (Teff)')
    ax.set_title('Error calibration by observing plan')
    ax.grid(alpha=0.3, axis='y')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8)
    _save(fig, 'fig4_by_plan.png')


def fig5():
    p = os.path.join(TABLES, 'leakage_summary.csv')
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    datasets = sorted(df['feature_set'].unique())
    targets = ['teff', 'logg', 'feh']
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 4.2),
                             squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        s = df[df['feature_set'] == ds].set_index('target').reindex(targets)
        x = np.arange(len(targets))
        w = 0.36
        ax.bar(x - w / 2, s['rmse_random'], w, label='random split (leaky)',
               color='#c53030')
        ax.bar(x + w / 2, s['rmse_grouped'], w, label='grouped split (correct)',
               color='#2b6cb0')
        for i, t in enumerate(targets):
            if pd.isna(s.loc[t, 'factor']):
                continue
            ax.text(i, max(s.loc[t, 'rmse_random'], s.loc[t, 'rmse_grouped']) * 1.02,
                    '%.2fx' % s.loc[t, 'factor'], ha='center', fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(targets)
        ax.set_title('features: %s' % ds)
        ax.set_ylabel('RMSE')
        ax.grid(alpha=0.3, axis='y')
        ax.legend(fontsize=8)
    fig.suptitle('Label leakage: random vs group-aware splitting', fontsize=11)
    _save(fig, 'fig5_leakage.png')


def fig6():
    p = os.path.join(TABLES, 'repeatability_snr_bins.csv')
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, param in zip(axes, ['teff', 'logg', 'feh']):
        s = df[df['value'] == param].sort_values('snr_lo')
        if s.empty:
            continue
        x = np.arange(len(s))
        ax.plot(x, s['ratio_sigma_over_err'], 'o-', color='#b7791f')
        ax.axhline(1.0, color='k', lw=1, ls='--')
        ax.set_xticks(x)
        ax.set_xticklabels(['%.0f' % v for v in s['snr_lo']], fontsize=7)
        ax.set_xlabel('median S/N bin (lower edge)')
        ax.set_ylabel('sigma / err')
        ax.set_title(param)
        ax.grid(alpha=0.3)
    fig.suptitle('Error under-estimation vs. signal-to-noise', fontsize=11)
    _save(fig, 'fig6_snr_bins.png')


def fig7():
    """容量扫描：泄漏倍数随模型容量与复观率变化。"""
    p = os.path.join(TABLES, 'leakage_capacity.csv')
    if not os.path.exists(p):
        return
    df = pd.read_csv(p)
    s = df[df['target'] == 'teff']
    if s.empty:
        return
    subsets = [x for x in ['all', 'repeat>=3', 'HIP', 'TD'] if x in set(s['subset'])]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    colors = {'all': '#2b6cb0', 'repeat>=3': '#276749', 'HIP': '#c53030',
              'TD': '#b7791f'}
    for sub in subsets:
        t = s[(s['subset'] == sub) & (s['model'] == 'lgbm')].copy()
        if t.empty:
            continue
        t['mcs'] = t['param'].str.extract(r'=(\d+)').astype(int)
        t = t.sort_values('mcs')
        hit = 100.0 * t['frac_exact_hit_random'].iloc[0]
        ax.plot(t['mcs'], t['factor'], 'o-', color=colors.get(sub, 'gray'),
                label='%s (repeat %.2fx, exact-hit %.0f%%)'
                      % (sub, t['n_samples'].iloc[0] / t['n_groups'].iloc[0], hit))
    ax.axhline(1.0, color='k', ls='--', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('min_child_samples  (smaller = higher capacity)')
    ax.set_ylabel('leakage factor  =  RMSE_grouped / RMSE_random')
    ax.set_title('Label leakage vs. model capacity and repeat rate  (target: Teff)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, 'fig7_leakage_capacity.png')


def fig8():
    """配对比较 vs 计划均值：表面差异几乎全是样本选择。"""
    p = os.path.join(TABLES, 'plan_means.csv')
    q = os.path.join(TABLES, 'plan_pair_offsets.csv')
    if not os.path.exists(p) or not os.path.exists(q):
        return
    means = pd.read_csv(p)
    pairs = pd.read_csv(q)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, param in zip(axes, ['teff', 'logg', 'feh']):
        m = means[means['target'] == param].sort_values('mean')
        if m.empty:
            continue
        pr = pairs[pairs['target'] == param]
        spread_means = float(m['mean'].max() - m['mean'].min())
        spread_pairs = float(pr['mean_diff'].max() - pr['mean_diff'].min()) if len(pr) else 0.0
        ax.barh(range(len(m)), m['mean'], color='#a0aec0', label='plan means (naive)')
        ax.annotate('spread of plan means = %.4g' % spread_means,
                    xy=(0.03, 0.94), xycoords='axes fraction', fontsize=8)
        ax.annotate('spread of PAIRED offsets = %.4g' % spread_pairs,
                    xy=(0.03, 0.87), xycoords='axes fraction', fontsize=8,
                    color='#c53030')
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m['plan_prefix'], fontsize=7)
        ax.set_title(param)
        ax.grid(alpha=0.3, axis='x')
        ax.legend(fontsize=7, loc='lower right')
    fig.suptitle('Why paired comparison matters: naive plan means are dominated by '
                 'target-population differences', fontsize=10)
    _save(fig, 'fig8_paired_vs_naive.png')


def main():
    print('=' * 70)
    print('  出图')
    print('=' * 70)
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print('  [%s 失败] %s' % (fn.__name__, exc))
    return 0


if __name__ == '__main__':
    sys.exit(main())
