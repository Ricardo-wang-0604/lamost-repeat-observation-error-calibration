# -*- coding: utf-8 -*-
"""
12_rv_frame.py —— 判定 LAMOST 波长网格是否在恒星静止系

为什么必须先判定
----------------
若波长已是恒星静止系（已做 RV 改正），再加一次改正就是**双重改正**，反而改错。

做法
----
取视向速度 |RV| 大的光谱，量强线实测线心位置对 RV 的回归斜率：

    未改正（观测系或日心系）：δλ = λ_rest · RV / c   → Hα 应为 0.021891 Å per km/s
    恒星静止系：δλ ≈ 0

**这个检验能证明什么、不能证明什么（重要，勿过度解读）**
-------------------------------------------------------
✅ 能证明：网格不是恒星静止系（否则斜率应为 0）
❌ 不能证明：网格是「观测系」还是「日心系」。
   若网格为日心系、而目录 rv 也是日心速度，线心同样位于 λ_rest(1+rv/c)，
   回归斜率**同样是 λ/c**，本检验无法区分。
   要区分必须把该次观测的日心速度 v_bary(RA, Dec, MJD) 放进二元回归。

⇒ 本项目只声称"未改正到恒星静止系"，并据此执行 RV 改正
  （两种坐标系假设下用目录 rv 做除法都能把线心移回静止波长）。

数据来源
--------
RV 取自目录表 `stellar.rv`（光谱 FITS 头里没有），按 OBSID 关联。

产出
----
results/tables/rv_frame_test.csv      逐谱线的实测线心与 RV
results/tables/rv_frame_report.txt    人读报告
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

C_KMS = 299792.458
RAW = os.path.join(ROOT, 'data', 'raw')
TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
LINES = [('Ha', 6562.80), ('Hb', 4861.33), ('Mgb', 5175.0), ('NaD', 5893.0)]
N_MAX = int(os.environ.get('RV_FRAME_N', '1200'))
LOG = []


def say(s=''):
    print(s, flush=True)
    LOG.append(str(s))


def line_center(wave, flux, lam0, half=12.0):
    """
    在 lam0±half 内用 (1-F/Fc) 的一阶矩求线心，Fc 取带内中位数。

    注意：该质心法对非对称谱线有偏置，因此**只用于斜率判定，不用于绝对波长定标**。
    （实测截距为正：Hα +1.755 Å、Hβ +3.073 Å、Mg b +2.158 Å，
     约 1.3 个 LRS 像元，来源未完全厘清。）
    """
    o = np.argsort(wave)
    w, f = wave[o], flux[o]
    sel = (w >= lam0 - half) & (w <= lam0 + half)
    if sel.sum() < 5:
        return np.nan
    ww, ff = w[sel], f[sel]
    fc = np.median(ff)
    if not np.isfinite(fc) or fc <= 0:
        return np.nan
    d = np.clip(1.0 - ff / fc, 0, None)      # 只看吸收
    if d.sum() <= 0:
        return np.nan
    return float(np.sum(ww * d) / np.sum(d))


def main():
    t0 = time.time()
    say('=' * 88)
    say('  LAMOST 波长坐标系判定（能否区分观测系与日心系）')
    say('=' * 88)

    # 解压缓存（由 06_spectra.py 建立）
    cache_root = os.path.join(RAW, 'spectra_20111024')
    files = []
    for r, _d, ns in os.walk(cache_root):
        for n in ns:
            if n.endswith('.fits.gz'):
                files.append(os.path.join(r, n))
    files.sort()
    say('解压缓存光谱: %d 条（本次用前 %d 条）' % (len(files), N_MAX))
    if not files:
        say('!! 无缓存。请先运行 06_spectra.py 建立解压缓存。')
        return 1

    rows = []
    for fp in files[:N_MAX]:
        try:
            sp = spectra.read_spectrum(fp)
        except Exception:  # noqa: BLE001
            continue
        h = sp.get('header', {})
        oid = h.get('OBSID')
        if oid is None or sp.get('wave') is None:
            continue
        rec = {'OBSID': int(oid)}
        for name, lam in LINES:
            rec['c_' + name] = line_center(sp['wave'], sp['flux'], lam)
        rows.append(rec)
    SF = pd.DataFrame(rows)
    say('成功测量 %d 条' % len(SF))

    cat = pd.read_pickle(os.path.join(ROOT, 'data', 'interim', 'stellar.pkl'))
    cat = cat[['obsid', 'rv', 'teff', 'snrr', 'subclass']].copy()
    cat['obsid'] = pd.to_numeric(cat['obsid'], errors='coerce')
    cat = cat.dropna(subset=['obsid'])
    cat['obsid'] = cat['obsid'].astype('int64')
    M = SF.merge(cat, left_on='OBSID', right_on='obsid', how='inner')
    M = M.dropna(subset=['rv'])
    say('关联到 RV 的: %d 条' % len(M))
    if len(M) < 30:
        say('!! 样本不足')
        return 1

    say('\nRV 分布: min=%.1f  中位=%.1f  max=%.1f km/s'
        % (M['rv'].min(), M['rv'].median(), M['rv'].max()))
    say('|RV| 分布: 中位=%.1f  P90=%.1f  max=%.1f km/s'
        % (M['rv'].abs().median(), M['rv'].abs().quantile(0.9),
           M['rv'].abs().max()))

    say('\n' + '=' * 88)
    say('线心偏移 vs RV —— 斜率判定')
    say('=' * 88)
    say('  %-6s %8s %10s %12s %12s %10s %10s'
        % ('谱线', 'n', 'λ_rest', '实测斜率', '预期 λ/c', '比值', 'r'))
    for name, lam in LINES:
        col = 'c_' + name
        s = M[[col, 'rv']].dropna()
        s = s[(s[col] > lam - 12) & (s[col] < lam + 12)]
        if len(s) < 30:
            say('  %-6s %8d  样本不足' % (name, len(s)))
            continue
        x = s['rv'].to_numpy(dtype=float)
        y = (s[col] - lam).to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        r = float(np.corrcoef(x, y)[0, 1])
        exp = lam / C_KMS
        say('  %-6s %8d %10.2f %12.6f %12.6f %10.3f %+10.3f'
            % (name, len(s), lam, slope, exp, slope / exp, r))
        say('        截距 %+.3f Å （质心法偏置，未完全厘清；只影响绝对定标不影响斜率）'
            % intercept)

    say('\n  判读：')
    say('    · 斜率显著非零 ⇒ 网格**不是恒星静止系**')
    say('    · 若网格为日心系而目录 rv 亦为日心速度，斜率**同样是 λ/c**，本检验无法区分')
    say('      ⇒ 只能声称"未改正到恒星静止系"，不能声称"确证为观测系"')
    say('    · Hβ 实测斜率约为 λ/c 的 1.27 倍（偏高），**未获解释**，如实记录')

    M.to_csv(os.path.join(TABLES, 'rv_frame_test.csv'), index=False,
             encoding='utf-8-sig')
    with open(os.path.join(TABLES, 'rv_frame_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    say('-> results/tables/rv_frame_test.csv, rv_frame_report.txt')
    return 0


if __name__ == '__main__':
    sys.exit(main())
