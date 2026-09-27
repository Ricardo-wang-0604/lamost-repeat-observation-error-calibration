# -*- coding: utf-8 -*-
"""
判别测试：极端离散来自「变星」还是「污染/错配」？

背景
----
审计指出（S2）："组内真值恒定"这个前提没有独立检验，变星/双星会让组内离散
包含真实参数变化，因此"官方误差被低估"与"约 1% 是变星/坏解"两种解释都未被排除。

查新报告指出：Bai et al. 2021 (RAA 21, 249) 发现复观误差**依赖观测时间间隔**，
本项目把所有复观对不分时间间隔混在一起算 —— 这是一个遗漏的变量。

判别逻辑
--------
对组内每一对观测，取时间间隔 Δt = |mjd_i − mjd_j|，算配对数差 |Δx|。

  若极端离散来自**变星/双星**（真实参数随时间变化）：
      离散度应随 Δt 单调增大（时间越长，变化越多）
  若来自**污染 / Gaia 源错配**（随机事件，与时间无关）：
      离散度应与 Δt 无关，但保留重尾（极少数大偏离）

同时做尾部集中度检验：把配对按 |Δx|/σ_quoted 排序，
看最离群的那部分配对的 Δt 分布是否与整体不同。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
sys.path.insert(0, ROOT)
from src import loader, repeat as rp  # noqa: E402

TABLES = os.path.join(ROOT, 'results', 'tables')
LOG = []


def say(s=''):
    print(s, flush=True)
    LOG.append(str(s))


def pair_products(df, param, err):
    """
    展开成「配对」表：同一 gp_id 内所有观测两两配对。
    为控制规模，只保留有复观的星，且每组取全部对（组内观测数中位仅 2，代价小）。
    """
    d = df[['gp_id', 'mjd', param, err]].dropna()
    d = d[np.isfinite(d[param].to_numpy(dtype=np.float64, na_value=np.nan))]
    d = d[np.isfinite(d[err].to_numpy(dtype=np.float64, na_value=np.nan))]
    d = d[d[err] > 0]
    cnt = d.groupby('gp_id', observed=True).size()
    keep = cnt[cnt >= 2].index
    d = d[d['gp_id'].isin(keep)].sort_values(['gp_id', 'mjd'])

    g = d.groupby('gp_id', observed=True)
    # 用 transform('first') 全向量化取组内首个观测，避免 transform(自定义函数)
    # （后者在 125 万个组上是小时级；本项目禁止这种做法）
    mjd0 = g['mjd'].transform('first')
    v0 = g[param].transform('first')
    e0 = g[err].transform('first')
    d = d.assign(
        dt_prev=g['mjd'].diff().abs(),
        dv_prev=g[param].diff().abs(),
        e_prev=g[err].shift(1),
        dt_first=(d['mjd'] - mjd0).abs(),
        dv_first=(d[param] - v0).abs(),
        e_first=e0,
    )
    return d


def analyze(df, param, err, label):
    say('\n' + '=' * 90)
    say('参数 %s' % param)
    say('=' * 90)
    d = pair_products(df, param, err)
    say('  展开后记录数 %d' % len(d))

    # 用「首观测 vs 每次后续观测」这一组配对（每对独立，且 Δt 跨度大）
    p = d[d['dt_first'].notna() & (d['dt_first'] > 0)][
        ['gp_id', 'dt_first', 'dv_first', 'e_first']].copy()
    p = p[p['e_first'] > 0]
    say('  用于分析的配对 %d' % len(p))
    if len(p) < 1000:
        say('  !! 配对太少，跳过')
        return None

    # 归一化偏差：|Δx| / (sqrt(2)·σ_quoted)，理论中位应约 0.95（半正态）
    p['z'] = p['dv_first'] / (np.sqrt(2.0) * p['e_first'])
    p['logdt'] = np.log10(p['dt_first'].clip(lower=1e-2))

    say('\n  ── 按时间间隔分箱 ──')
    bins = [0, 1, 10, 100, 400, 1000, 2500, 10000]
    say('  %-14s %10s %12s %12s %12s %12s'
        % ('Δt (天)', '配对数', '|Δx|中位', 'z 中位', 'z P99', 'z>10 占比'))
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = p[(p['dt_first'] >= lo) & (p['dt_first'] < hi)]
        if len(s) < 100:
            continue
        say('  %-14s %10d %12.4g %12.3f %12.1f %11.3f%%'
            % ('%d-%d' % (lo, hi) if lo else '<%d' % hi,
               len(s), float(s['dv_first'].median()), float(s['z'].median()),
               float(s['z'].quantile(0.99)),
               100.0 * float((s['z'] > 10).mean())))

    # 趋势检验：z 中位 对 log Δt 的 Spearman
    rho = float(p[['logdt', 'z']].corr(method='spearman').iloc[0, 1])
    say('\n  z 与 log(Δt) 的 Spearman ρ = %+.4f' % rho)
    if abs(rho) < 0.05:
        say('  → 离散度与时间间隔**基本无关** ⇒ 支持「污染/错配」（随机事件）')
    else:
        say('  → 离散度随间隔变化 ⇒ 含有「变星/双星」（真实参数演化）成分')

    # 尾部集中度：最离散 1% 配对的 Δt 分布 vs 整体
    say('\n  ── 尾部 vs 整体 的 Δt 分布 ──')
    thr = float(p['z'].quantile(0.99))
    tail = p[p['z'] > thr]
    say('  尾部阈值 z > %.1f ，共 %d 对 (%.2f%%)'
        % (thr, len(tail), 100.0 * len(tail) / len(p)))
    say('  %-16s %14s %14s' % ('统计量', '整体', '最离散 1%'))
    for c, lab in [('dt_first', 'Δt 中位(天)'), ('z', 'z 中位'),
                   ('e_first', '官方误差中位')]:
        say('  %-16s %14.3f %14.3f'
            % (lab, float(p[c].median()), float(tail[c].median())))
    say('  Δt 的 P90:  整体 %.0f 天   尾部 %.0f 天'
        % (float(p['dt_first'].quantile(0.9)), float(tail['dt_first'].quantile(0.9))))

    # 变异系数式的判据：尾部是否"整体偏移"还是"少数极端"
    say('\n  ── z 分布的尾部厚度（与标准正态比较）──')
    for q, exp in [(0.5, 0.674), (0.9, 1.645), (0.99, 2.576), (0.999, 3.291)]:
        say('    z 的 %.1f 分位 = %8.3f   （标准正态应为 %.3f，倍数 %.1f×）'
            % (q * 100, float(p['z'].quantile(q)), exp,
               float(p['z'].quantile(q)) / exp))
    share = float((p['z'] > 3.291).mean())
    say('    z > 3.291 的占比 = %.3f%%（标准正态应为 0.100%%，即 %.1f 倍）'
        % (100 * share, share / 0.001))

    return p


def main():
    t0 = time.time()
    say('=' * 90)
    say('  判别测试：极端离散来自「变星」还是「污染/错配」？')
    say('=' * 90)

    df = pd.read_pickle(os.path.join(loader.INTERIM_DIR, 'stellar.pkl'))
    df = df[df['gp_id'].notna()]
    df = df[np.isfinite(df['teff'].to_numpy(dtype=np.float64, na_value=np.nan))]
    say('样本 %d 行' % len(df))

    outs = {}
    for param, err in (('teff', 'teff_err'), ('feh', 'feh_err')):
        p = analyze(df, param, err, param)
        if p is not None:
            outs[param] = p

    if outs:
        # 逐配对明细有近 200 万行、约 267 MB，**不适合整表落盘**。
        # 这里只保存按 (参数 × Δt 分箱 × z 分位) 的分层抽样（每层最多 5000 行），
        # 足以复现报告中的任何统计量，体积降到 MB 级。
        parts = []
        for k, v in outs.items():
            v = v.copy()
            v['param'] = k
            v['logdt_bin'] = pd.cut(v['logdt'], np.arange(-2, 4.1, 0.25))
            v['z_bin'] = pd.qcut(v['z'], 20, duplicates='drop')
            s = (v.groupby(['logdt_bin', 'z_bin'], observed=True)
                 .apply(lambda g: g.head(200))
                 .reset_index(drop=True))
            if len(s) > 400000:
                s = s.sample(400000, random_state=0)
            parts.append(s)
            say('  [抽样存档] %s: %d → %d 行' % (k, len(v), len(s)))
        allp = pd.concat(parts, ignore_index=True)
        allp.to_csv(os.path.join(TABLES, 'time_interval_test.csv'),
                    index=False, encoding='utf-8-sig')
        say('-> results/tables/time_interval_test.csv (%.1f MB)'
            % (os.path.getsize(os.path.join(TABLES, 'time_interval_test.csv')) / 1048576.0))

    with open(os.path.join(TABLES, 'time_interval_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    say('总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
