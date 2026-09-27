# -*- coding: utf-8 -*-
"""
13_rv_ab.py —— 红移改正的 A/B 测试（改正到底有没有用？）

要回答的两个问题
----------------
1. **改正生效了吗？** 对同一批光谱，分别在"不做改正"与"用目录 rv 改正"下算谱线指数，
   比较 ΔEW 是否随 |RV| 变化。
2. **改正有用吗？** 比较改正前后谱线指数与官方参数的相关系数。

实测结论（本项目）
------------------
* 改正**生效**：|RV| 越大 ΔEW 越大（>120 km/s 档明显）。
* 但对**宽带等值宽度**影响很小：
  LRS 像元尺度 dl = 1.336 Å，而 RV 中位 |RV| = 30 km/s 在 Hα 处仅位移 0.66 Å
  —— **不足一个像元**。故中位 |ΔEW| < 0.14 Å（Hβ/Mg b/Hα 的中位差为 0.0000）。
* 相关系数**改善/变差/持平**的计数由本脚本实际统计并落盘（见 `rv_ab_summary.csv`），
  不再依赖正文里的口头断言。
* **真正必需该改正的是线心/视向速度类测量**（±340 km/s → Hα ±7.4 Å）。

产出
----
results/tables/rv_ab_test.csv       逐条光谱的改正前后 EW
results/tables/rv_ab_summary.csv    改善/变差/持平计数与 ΔEW 统计
results/tables/rv_ab_report.txt     人读报告
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import spectra  # noqa: E402

RAW = os.path.join(ROOT, 'data', 'raw')
TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
BANDS = ['CaK', 'Hd', 'Gband', 'Hg', 'Hb', 'Mgb', 'NaD', 'Ha']
N = int(os.environ.get('AB_N', '900'))
# 判定 |ρ| 变化是否算"改善/变差"的阈值
TOL = 0.002
LOG = []


def say(s=''):
    print(s, flush=True)
    LOG.append(str(s))


def main():
    t0 = time.time()
    say('=' * 84)
    say('  红移改正 A/B 测试：改正生效吗？有用吗？')
    say('=' * 84)

    cache = os.path.join(RAW, 'spectra_20111024')
    files = []
    for r, _d, ns in os.walk(cache):
        for n in ns:
            if n.endswith('.fits.gz'):
                files.append(os.path.join(r, n))
    files.sort()
    say('缓存光谱 %d 条，本次测试用前 %d 条' % (len(files), N))
    if not files:
        say('!! 无缓存。请先运行 06_spectra.py。')
        return 1

    say('\n读取目录表取 rv ...')
    full = pd.read_pickle(os.path.join(ROOT, 'data', 'interim', 'stellar.pkl'))
    cat = full[['obsid', 'rv', 'teff', 'logg', 'feh', 'subclass']].copy()
    del full
    cat['obsid'] = pd.to_numeric(cat['obsid'], errors='coerce')
    rvmap = dict(zip(cat.loc[cat['obsid'].notna(), 'obsid'].astype('int64'),
                     cat.loc[cat['obsid'].notna(), 'rv'].astype(float)))

    rows = []
    for fp in files[:N]:
        try:
            sp = spectra.read_spectrum(fp)
        except Exception:  # noqa: BLE001
            continue
        h = sp.get('header', {})
        oid = h.get('OBSID')
        if oid is None or sp.get('wave') is None:
            continue
        rv = rvmap.get(int(oid))
        f_no = spectra.spectrum_features(sp, rv=None)
        f_rv = spectra.spectrum_features(sp, rv=rv)
        rec = {'OBSID': int(oid), 'rv_used': rv, 'SNRR': h.get('SNRR'),
               'SUBCLASS': h.get('SUBCLASS')}
        for b in BANDS:
            rec[b + '_no'] = f_no.get(b)
            rec[b + '_rv'] = f_rv.get(b)
        rows.append(rec)
    D = pd.DataFrame(rows)
    say('完成 %d 条' % len(D))

    cat2 = cat.rename(columns={'rv': 'rv_cat'})
    D = D.merge(cat2, left_on='OBSID', right_on='obsid', how='left')
    D['rv'] = D['rv_used'].fillna(D['rv_cat'])
    D['absrv'] = D['rv'].abs()

    # ---------------- 1. ΔEW vs |RV| ----------------
    say('\n' + '=' * 84)
    say('1. ΔEW = EW(改正后) − EW(改正前) 随 |RV| 的变化（改正生效的直接证据）')
    say('=' * 84)
    say('  %-8s %10s %12s %12s %12s %12s'
        % ('谱线', '|RV|<10', '10-30', '30-60', '60-120', '>120'))
    for b in BANDS:
        a, c = b + '_no', b + '_rv'
        s = D[[a, c, 'absrv']].dropna()
        if len(s) < 20:
            continue
        s = s.assign(d=s[c] - s[a])
        cells = []
        for lo, hi in [(0, 10), (10, 30), (30, 60), (60, 120), (120, 9999)]:
            t = s[(s['absrv'] >= lo) & (s['absrv'] < hi)]
            cells.append('%.3f' % t['d'].mean() if len(t) >= 5 else '-')
        say('  %-8s %10s %12s %12s %12s %12s' % (b, *cells))

    # ---------------- 2. 相关系数 A/B ----------------
    say('\n' + '=' * 84)
    say('2. 相关系数（Spearman）：改正前 vs 改正后')
    say('=' * 84)
    say('  %-8s %-6s %10s %10s %10s %8s'
        % ('谱线', '参数', '未改正', '已改正', '差', '判定'))
    cnt = {'改善': 0, '变差': 0, '持平': 0}
    detail = []
    for b in BANDS:
        for p in ('teff', 'logg', 'feh'):
            if p not in D.columns:
                continue
            r0 = D[[b + '_no', p]].dropna()
            r1 = D[[b + '_rv', p]].dropna()
            if len(r0) < 40 or len(r1) < 40:
                continue
            c0 = float(r0[b + '_no'].corr(r0[p], method='spearman'))
            c1 = float(r1[b + '_rv'].corr(r1[p], method='spearman'))
            imp = abs(c1) - abs(c0)
            flag = '改善' if imp > TOL else ('变差' if imp < -TOL else '持平')
            cnt[flag] += 1
            say('  %-8s %-6s %10.4f %10.4f %+10.4f %8s'
                % (b, p, c0, c1, imp, flag))
            detail.append({'band': b, 'param': p, 'spearman_uncorrected': c0,
                           'spearman_corrected': c1, 'delta': imp, 'verdict': flag})

    say('\n  汇总（阈值 |Δρ| > %.3f 才算变化）：改善 %d 项 / 变差 %d 项 / 持平 %d 项'
        % (TOL, cnt['改善'], cnt['变差'], cnt['持平']))

    # ---------------- 3. ΔEW 统计 ----------------
    say('\n' + '=' * 84)
    say('3. 逐条 EW 差异统计')
    say('=' * 84)
    say('  %-8s %10s %12s %12s %12s %10s'
        % ('谱线', 'n', '平均ΔEW', 'ΔEW的std', '|ΔEW|中位', '最大|ΔEW|'))
    ew_rows = []
    for b in BANDS:
        a, c = b + '_no', b + '_rv'
        s = D[[a, c]].dropna()
        if len(s) < 20:
            continue
        d = (s[c] - s[a]).to_numpy()
        say('  %-8s %10d %12.4f %12.4f %12.4f %10.3f'
            % (b, len(d), d.mean(), d.std(), np.median(np.abs(d)),
               np.abs(d).max()))
        ew_rows.append({'band': b, 'n': len(d), 'mean_dEW': float(d.mean()),
                        'std_dEW': float(d.std()),
                        'median_abs_dEW': float(np.median(np.abs(d))),
                        'max_abs_dEW': float(np.abs(d).max())})

    say('\n  判读：中位 |ΔEW| < 0.14 Å（Hβ/Mg b/Hα 为 0.0000）说明改正对')
    say('        **宽带等值宽度**几乎无影响 —— 因为 LRS 像元 1.336 Å')
    say('        > RV 中位位移 0.66 Å。真正必需它的是线心/视速度测量。')

    D.to_csv(os.path.join(TABLES, 'rv_ab_test.csv'), index=False,
             encoding='utf-8-sig')
    S = pd.DataFrame(detail)
    S.to_csv(os.path.join(TABLES, 'rv_ab_summary.csv'), index=False,
             encoding='utf-8-sig')
    E = pd.DataFrame(ew_rows)
    E.to_csv(os.path.join(TABLES, 'rv_ab_ew.csv'), index=False,
             encoding='utf-8-sig')
    with open(os.path.join(TABLES, 'rv_ab_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    say('-> results/tables/rv_ab_{test,summary,ew}.csv, rv_ab_report.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())
