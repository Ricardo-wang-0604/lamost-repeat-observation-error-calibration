# -*- coding: utf-8 -*-
"""
08_report.py —— 把所有结果表汇总成一份可直接阅读的 Markdown 报告

只做汇总与排版，不重新计算。所有数字都来自 results/tables/ 下的 CSV，
因此报告与数据严格一致、可追溯。
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
TABLES = os.path.join(ROOT, 'results', 'tables')
REPORT = os.path.join(ROOT, 'results', 'REPORT.md')


def rd(name):
    p = os.path.join(TABLES, name)
    if not os.path.exists(p):
        return None
    try:
        return pd.read_csv(p)
    except Exception:  # noqa: BLE001
        return None


def read_text(path):
    """
    宽容地读文本。

    PowerShell 的 `Out-File -Encoding utf8` 在 Windows PowerShell 下写出的是
    **UTF-16 LE 带 BOM**（首字节 0xff），而本项目的报告文件是 utf-8-sig。
    两种都要能读，否则汇总会直接崩。
    """
    with open(path, 'rb') as f:
        raw = f.read()
    for enc in ('utf-8-sig', 'utf-16', 'utf-8', 'gbk'):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode('utf-8', 'replace')


def md_table(df, cols=None, floatfmt='%.4f', maxrows=None):
    if df is None or df.empty:
        return '_（无数据）_\n'
    d = df.copy()
    if cols:
        d = d[[c for c in cols if c in d.columns]]
    if maxrows:
        d = d.head(maxrows)
    hdr = '| ' + ' | '.join(str(c) for c in d.columns) + ' |'
    sep = '|' + '|'.join(['---'] * len(d.columns)) + '|'
    lines = [hdr, sep]
    for _, r in d.iterrows():
        cells = []
        for v in r:
            if isinstance(v, float):
                cells.append('nan' if not np.isfinite(v) else (floatfmt % v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines) + '\n'


def main():
    L = []
    A = L.append
    A('# LAMOST DR10 复观恒星：官方误差标定与机器学习标签泄漏\n')
    A('自动生成于 %s\n' % time.strftime('%Y-%m-%d %H:%M:%S'))

    # ---------- 1. 规模 ----------
    A('\n## 1. 数据规模\n')
    L2 = []
    for fn in ('qc_report.txt', 'repeatability_report.txt',
               'repeatability_by_plan.csv', 'leakage_capacity.log'):
        p = os.path.join(TABLES, fn)
        if os.path.exists(p):
            L2.append((fn, read_text(p)))

    import re
    scale = {}
    for fn, txt in L2:
        for key, pat in [('obs', r'观测总数\s+(\d+)'),
                         ('gp', r'唯一 gp_id\s+(\d+)'),
                         ('des', r'唯一 designation\s+(\d+)')]:
            m = re.search(pat, txt)
            if m and key not in scale:
                scale[key] = int(m.group(1))
    A('| 项 | 数值 |')
    A('|---|---|')
    A('| 观测总数 | %s |' % f"{scale.get('obs', 0):,}")
    A('| 唯一恒星（gp_id） | %s |' % f"{scale.get('gp', 0):,}")
    A('| 唯一位置编码（designation） | %s |' % f"{scale.get('des', 0):,}")
    if scale.get('gp'):
        A('| 复观倍率 | %.3f× |' % (scale['obs'] / scale['gp']))

    # ---------- 2. 总体标定 ----------
    A('\n## 2. 复观一致性与官方误差标定（总体）\n')
    ov = rd('repeatability_overall.csv')
    if ov is not None:
        cols = ['value', 'n_groups', 'n_obs', 'pooled_sigma', 'err_median',
                'ratio_sigma_over_err', 'chi2_red']
        A(md_table(ov, [c for c in cols if c in ov.columns]))
        A('\n- `pooled σ`：只由复观离散度决定的实测不确定度。')
        A('- `chi2_red`：官方误差若正确应为 1。')
        A('- `σ/err`：实测离散与官方误差之比，>1 表示官方误差偏小。\n')

    # ---------- 3. 按计划 ----------
    A('\n## 3. 按观测计划分层\n')
    bp = rd('repeatability_by_plan.csv')
    if bp is not None and 'value' in bp.columns:
        for param in ('teff', 'logg', 'feh'):
            s = bp[bp['value'] == param]
            if s.empty:
                continue
            s = s.sort_values('n_groups', ascending=False)
            A('\n### %s\n' % param)
            cols = ['plan_prefix', 'n_groups', 'pooled_sigma', 'err_median',
                    'ratio_sigma_over_err', 'chi2_red']
            A(md_table(s, [c for c in cols if c in s.columns]))

    # ---------- 4. S/N 与次数 ----------
    A('\n## 4. 不确定度随信噪比与复观次数的变化\n')
    gs = rd('repeatability_groupsize.csv')
    if gs is not None:
        A('\n### 按复观次数\n')
        A(md_table(gs, ['value', 'n_obs_bin', 'n_stars', 'pooled_sigma',
                        'err_median', 'ratio_sigma_over_err', 'chi2_red']))
    sb = rd('repeatability_snr_bins.csv')
    if sb is not None:
        A('\n### 按信噪比分箱\n')
        A(md_table(sb, ['value', 'snr_lo', 'snr_hi', 'n_groups', 'pooled_sigma',
                        'err_median', 'ratio_sigma_over_err', 'chi2_red']))

    # ---------- 5. 泄漏 ----------
    A('\n## 5. 机器学习标签泄漏\n')
    ls = rd('leakage_summary.csv')
    if ls is not None:
        cols = ['feature_set', 'target', 'features', 'n_samples', 'n_groups',
                'rmse_random', 'rmse_grouped', 'factor', 'rel_inflation_pct',
                'r2_random', 'r2_grouped']
        A(md_table(ls, [c for c in cols if c in ls.columns]))
        A('\n- `factor = RMSE_grouped / RMSE_random`，>1 表示随机划分把 RMSE 压低了。')
        A('- `feature_set = photometry` 的特征是恒星级常数，泄漏最严重。\n')
    else:
        A('_（泄漏实验尚未完成）_\n')

    # ---------- 6. 跨计划 ----------
    A('\n## 6. 跨观测计划的系统偏差（配对）\n')
    po = rd('plan_pair_offsets.csv')
    if po is not None and not po.empty:
        for param in ('teff', 'logg', 'feh'):
            s = po[po['target'] == param].sort_values('n_common', ascending=False)
            if s.empty:
                continue
            A('\n### %s\n' % param)
            A(md_table(s, ['plan_a', 'plan_b', 'n_common', 'mean_diff',
                           'std_diff', 'sem', 't_stat'], maxrows=15))
    else:
        A('_（尚未计算）_\n')

    # ---------- 7. 光谱 ----------
    A('\n## 7. 真实光谱特征\n')
    sc = rd('spectra_correlations.csv')
    if sc is not None and not sc.empty:
        piv = sc.pivot_table(index='band', columns='param', values='spearman')
        piv = piv.reset_index()
        # 不用 DataFrame.to_markdown —— 它依赖未安装的 tabulate。
        A(md_table(piv, floatfmt='%+.3f'))
        ns = sc.groupby('band')['n'].max().to_dict()
        A('\n各谱线可用的最大匹配数: ' +
          ', '.join('%s=%d' % (k, v) for k, v in sorted(ns.items())))
    else:
        A('_（尚未计算）_\n')

    txt = '\n'.join(L)
    with open(REPORT, 'w', encoding='utf-8-sig') as f:
        f.write(txt)
    print('-> %s (%d 字符)' % (REPORT, len(txt)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
