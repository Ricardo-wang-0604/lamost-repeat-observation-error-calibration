# -*- coding: utf-8 -*-
"""
11_outliers.py —— 刻画主导 χ² 的极端异常组：它们是什么天体？

为什么需要这一步
----------------
`10_robustness.py` 证明全样本的 χ²_red = 5.875 由约 1% 的组主导。
但"由少数组主导"只能说明结论不稳健，**还不能说明那些组是污染**。
必须刻画它们，才能区分：

  (A) 目标本身参数在变（变星 / 双星）
  (B) 污染 / Gaia 源错配（把不同天体当成同一颗星）

本脚本给出画像；**判别测试见 `09_time_interval.py`**
（变星会让离散度随观测时间间隔增大，污染不会）。

实现要点
--------
慢的做法是对 125 万个组做 `mode()` 聚合（125 万次 Python 调用，小时级）。
这里先向量化算出逐组 χ²/dof（秒级），**只对被抽样的组**再取观测级属性。

产出
----
results/tables/outlier_bad_sample.csv    χ²/dof > P99 的抽样组
results/tables/outlier_good_sample.csv   正常抽样组（对照）
results/tables/outlier_report.txt        人读报告
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


def main():
    t0 = time.time()
    say('=' * 88)
    say('  极端异常组的性质（它们是什么天体？）')
    say('=' * 88)

    df = pd.read_pickle(os.path.join(loader.INTERIM_DIR, 'stellar.pkl'))
    df = df[df['gp_id'].notna()]
    df = df[np.isfinite(df['teff'].to_numpy(dtype=np.float64, na_value=np.nan))]
    say('样本 %d 行' % len(df))

    st = rp.group_stats(df, key='gp_id', value='teff', err='teff_err', min_obs=2)
    st = st[st['chi2'].notna() & (st['dof'] > 0)].copy()
    st['chi2_dof'] = st['chi2'] / st['dof']
    say('复观组 %d 个' % len(st))

    # ---------------- 1. χ² 的集中度 ----------------
    say('\n' + '-' * 88)
    say('一、χ² 的集中度（"少数组主导"的量化）')
    say('-' * 88)
    tot = float(st['chi2'].sum())
    s2 = st.sort_values('chi2', ascending=False)
    for frac, lab in ((0.0001, '0.01%'), (0.001, '0.1%'), (0.003, '0.3%'),
                      (0.01, '1%'), (0.05, '5%')):
        k = max(1, int(round(len(s2) * frac)))
        share = float(s2['chi2'].iloc[:k].sum()) / tot
        say('  最离散的 %-6s 组（%9d 个）贡献全部 χ² 的 %5.1f%%'
            % (lab, k, share * 100))

    say('\n  χ²/dof 分位:')
    for q in (0.5, 0.9, 0.99, 0.999, 0.9999):
        say('    %7.2f%%   %12.2f' % (q * 100, float(st['chi2_dof'].quantile(q))))
    say('    %7s   %12.1f' % ('max', float(st['chi2_dof'].max())))

    # ---------------- 2. 抽样后的逐组属性 ----------------
    say('\n' + '-' * 88)
    say('二、抽样组画像（异常组 vs 正常组）')
    say('-' * 88)
    thr = float(st['chi2_dof'].quantile(0.99))
    bad_idx = st.index[st['chi2_dof'] > thr]
    good_idx = st.index[st['chi2_dof'] <= thr]

    rng = np.random.RandomState(0)
    bad_s = pd.Index(rng.choice(bad_idx, size=min(12000, len(bad_idx)), replace=False))
    good_s = pd.Index(rng.choice(good_idx, size=min(12000, len(good_idx)), replace=False))
    say('  异常组定义 χ²/dof > P99 = %.1f，共 %d 个 (%.2f%%)'
        % (thr, len(bad_idx), 100.0 * len(bad_idx) / len(st)))
    say('  抽样: 异常 %d 组，正常 %d 组' % (len(bad_s), len(good_s)))

    def profile(idx, label):
        sub = df[df['gp_id'].isin(idx)]
        g = sub.groupby('gp_id', observed=True)
        out = pd.DataFrame({
            'n_obs': g.size(),
            'teff_med': g['teff'].median(),
            'teff_min': g['teff'].min(),
            'teff_max': g['teff'].max(),
            'snr_med': g['snrr'].median(),
        })
        out['teff_range'] = out['teff_max'] - out['teff_min']
        out = out.join(st.loc[out.index, ['chi2_dof', 'err_median']])
        say('\n  [%s]  %d 组' % (label, len(out)))
        say('    %-18s %14s %14s' % ('量', '中位', 'P90'))
        for c, lab in [('n_obs', '复观次数'), ('teff_med', 'Teff中位(K)'),
                       ('teff_range', 'Teff组内极差(K)'), ('snr_med', 'SNR中位'),
                       ('err_median', '官方误差中位(K)'), ('chi2_dof', 'χ²/dof')]:
            if c in out.columns:
                say('    %-18s %14.2f %14.2f'
                    % (lab, float(out[c].median()), float(out[c].quantile(0.9))))
        rr = out['teff_range'] / out['err_median'].replace(0, np.nan)
        say('    %-18s %14.2f %14.2f   ← 极差/官方误差（正常应约 1-3）'
            % ('极差比', float(rr.median()), float(rr.quantile(0.9))))
        for col, nm in (('subclass', '光谱型'), ('plan_prefix', '计划')):
            if col in sub.columns:
                vc = sub[col].value_counts().head(8)
                tot_s = len(sub)
                say('    %s分布: %s' % (nm, ', '.join(
                    '%s=%.1f%%' % (str(k), 100.0 * v / tot_s) for k, v in vc.items())))
        return out

    B = profile(bad_s, '异常组')
    G = profile(good_s, '正常组')

    # ---------------- 3. 极端 20 例 ----------------
    say('\n' + '-' * 88)
    say('三、χ²/dof 最大的 20 个组')
    say('-' * 88)
    top = st.sort_values('chi2_dof', ascending=False).head(20).index
    sub = df[df['gp_id'].isin(top)]
    g = sub.groupby('gp_id', observed=True)
    T = pd.DataFrame({'n_obs': g.size(), 'teff_min': g['teff'].min(),
                      'teff_max': g['teff'].max(), 'snr_med': g['snrr'].median(),
                      'sub': g['subclass'].first(), 'plan': g['plan_prefix'].first()})
    T['range'] = T['teff_max'] - T['teff_min']
    T = T.join(st.loc[top, ['chi2_dof', 'err_median']]).sort_values(
        'chi2_dof', ascending=False)
    say('  %-22s %5s %11s %9s %9s %8s %8s %-7s %-6s'
        % ('gp_id', 'n', 'χ²/dof', 'Tmin', 'Tmax', '极差', 'SNR', '光谱型', '计划'))
    for gid, r in T.iterrows():
        say('  %-22s %5d %11.1f %9.1f %9.1f %8.1f %8.1f %-7s %-6s'
            % (str(gid)[:22], r['n_obs'], r['chi2_dof'], r['teff_min'],
               r['teff_max'], r['range'], r['snr_med'],
               str(r['sub'])[:7], str(r['plan'])[:6]))

    say('\n  判读：若极差远大于官方误差（例如差 10 倍以上），')
    say('        那不是"误差公式偏小"，而是**该组的多次观测根本不是同一颗星**')
    say('        （光纤串扰/污染、把不同天体类型当成恒星拟合），')
    say('        或者**目标本身参数在变**（变星、双星）。')
    say('        两者靠 `09_time_interval.py` 的时间间隔判别测试区分。')

    B.to_csv(os.path.join(TABLES, 'outlier_bad_sample.csv'), encoding='utf-8-sig')
    G.to_csv(os.path.join(TABLES, 'outlier_good_sample.csv'), encoding='utf-8-sig')
    with open(os.path.join(TABLES, 'outlier_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    say('-> results/tables/outlier_report.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())
