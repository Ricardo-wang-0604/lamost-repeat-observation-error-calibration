# -*- coding: utf-8 -*-
"""
06_spectra.py —— LAMOST 真实光谱的特征提取与目录参数交叉验证

流程
----
  1. 流式遍历日包 20111024.tar.gz（9817 条光谱），提取谱线等值宽度与连续谱斜率
  2. 按 FITS 头里的 OBSID 与目录表 stellar 精确 join
  3. 检验谱线指数与官方 Teff / logg / [Fe/H] 的相关性
     —— 若谱线指数确实携带恒星参数信息，相关关系应当显著且方向正确
  4. 用光谱特征复现标签泄漏实验（复观部分）

为什么要 join
-------------
FITS 头里的 OBSID 与 stellar.obsid 是同一编号，因此可以把"我自己从光谱量出来的"
与"官方管线给出的"放在同一行比较 —— 这是验证光谱处理是否正确的关键。

产出
----
data/interim/spectra_features.csv        9817 条光谱的特征
results/tables/spectra_join.csv          与目录 join 后的表
results/tables/spectra_correlations.csv  谱线指数 × 官方参数 的相关系数
results/tables/spectra_report.txt
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = r'D:\ds工作区\01-科研实习\LAMOST-复观恒星'
sys.path.insert(0, ROOT)
from src import loader, spectra  # noqa: E402

RAW = os.path.join(ROOT, 'data', 'raw')
TABLES = os.path.join(ROOT, 'results', 'tables')
os.makedirs(TABLES, exist_ok=True)
LOG = []
BAND_COLS = ['CaK', 'Hd', 'Gband', 'Hg', 'Hb', 'Mgb', 'NaD', 'Ha']


def say(s=''):
    print(s)
    LOG.append(str(s))


def ensure_extracted(tar_path, outdir):
    """
    把日包**一次性顺序解压**到磁盘，返回解压出的 .fits.gz 路径列表。

    为什么必须先解压
    ----------------
    `tarfile.open(tar, 'r:gz')` 是不可寻址的流式读取。astropy 打开 FITS 时会做
    随机寻址（seek），而底层 GzipFile 无法 seek —— 于是**每次都从日包开头
    重新解压一遍**，复杂度退化到 O(N²)。实测每条耗时随条数增长：

        N=100 → 0.022 s/条      N=400 → 0.200 s/条

    按此推算 9817 条需要 5 小时以上。而先解压一次是 O(N) 单次遍历，
    之后 astropy 面对的是可正常 seek 的真实文件。
    """
    if os.path.isdir(outdir):
        files = []
        for root, _dirs, names in os.walk(outdir):
            for n in names:
                if n.endswith('.fits.gz'):
                    files.append(os.path.join(root, n))
        if files:
            say('  解压缓存已存在: %d 个文件' % len(files))
            return sorted(files)

    import tarfile
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    say('  一次性解压日包到 %s ...' % outdir)
    with tarfile.open(tar_path, 'r:gz') as tf:
        tf.extractall(outdir)
    files = []
    for root, _dirs, names in os.walk(outdir):
        for n in names:
            if n.endswith('.fits.gz'):
                files.append(os.path.join(root, n))
    say('  解压完成: %d 个文件，用时 %.1f 分'
        % (len(files), (time.time() - t0) / 60.0))
    return sorted(files)


def extract(limit_per_tar=None, rvmap=None):
    """
    读取光谱并提取谱线特征。先解压再逐条读，避免 O(N²) 退化。

    rvmap: {obsid -> rv(km/s)}，来自目录表 stellar。
           用于把观测系波长换算到恒星静止系（见 src/spectra.py 的 WAVELENGTH_FRAME）。
    """
    tars = sorted([p for p in os.listdir(RAW) if p.endswith('.tar.gz')])
    if not tars:
        say('!! data/raw 下没有 .tar.gz 日包')
        return None
    say('日包: %s' % ', '.join(tars))
    if limit_per_tar:
        say('限量 %d 条/日包' % limit_per_tar)
    say('红移改正: %s' % ('启用（rv 来自目录表）' if rvmap else '关闭（无 rv 映射）'))

    rows, errs = [], []
    n_rv = 0

    def on_err(name, exc):
        if len(errs) < 10:
            errs.append((name, repr(exc)[:120]))

    t0 = time.time()
    for tar in tars:
        p = os.path.join(RAW, tar)
        cache = os.path.join(RAW, 'spectra_' + tar.replace('.tar.gz', ''))
        say('\n处理 %s (%.1f MB)' % (tar, os.path.getsize(p) / 1048576.0))
        files = ensure_extracted(p, cache)
        if limit_per_tar:
            files = files[:limit_per_tar]
        say('  待处理 %d 条' % len(files))

        n = 0
        for fp in files:
            try:
                sp = spectra.read_spectrum(fp)
            except Exception as exc:  # noqa: BLE001
                on_err(fp, exc)
                continue
            sp['name'] = fp
            sp['parsed'] = spectra.parse_name(fp)
            oid = sp.get('header', {}).get('OBSID')
            rv = None
            if rvmap is not None and oid is not None:
                try:
                    rv = rvmap.get(int(oid))
                except (TypeError, ValueError):
                    rv = None
                if rv is not None:
                    n_rv += 1
            rows.append(spectra.spectrum_features(sp, rv=rv))
            n += 1
            if n % 500 == 0:
                say('  已处理 %d/%d，%.1f 分' % (n, len(files),
                                                 (time.time() - t0) / 60.0))
        say('  %s 完成: %d 条，累计 %.1f 分'
            % (tar, n, (time.time() - t0) / 60.0))

    if errs:
        say('\n读取失败样例 (%d 条):' % len(errs))
        for nme, e in errs[:5]:
            say('  %s -> %s' % (nme, e))

    df = pd.DataFrame(rows)
    say('\n合计 %d 条光谱，其中 %d 条取得 RV 用于红移改正，耗时 %.1f 分'
        % (len(df), n_rv, (time.time() - t0) / 60.0))
    return df


def main():
    t0 = time.time()
    say('=' * 84)
    say('  LAMOST 真实光谱特征提取与目录参数交叉验证')
    say('=' * 84)

    full = '--full' in sys.argv
    limit = None if full else (int(os.environ.get('LAMOST_SPECTRA_LIMIT', '3000')))
    if limit:
        say('模式: 限量 %d 条（加 --full 可尝试全量）' % limit)

    # ---------------- 先读目录，建立 obsid -> rv 映射 ----------------
    # 红移改正需要 rv，而 rv 只在目录表 stellar 里（光谱 FITS 头没有）。
    # 因此必须**先**读目录、**再**提取光谱特征，顺序不能颠倒。
    pkl = os.path.join(loader.INTERIM_DIR, 'stellar.pkl')
    rvmap = None
    cat = None
    if os.path.exists(pkl):
        say('\n读取目录表以获取视向速度 ...')
        _full = pd.read_pickle(pkl)
        cat = _full[[c for c in ['obsid', 'gp_id', 'teff', 'teff_err', 'logg',
                                 'feh', 'feh_err', 'snrr', 'rv', 'subclass',
                                 'plan_prefix'] if c in _full.columns]].copy()
        del _full
        cat['obsid'] = pd.to_numeric(cat['obsid'], errors='coerce')
        rvmap = {}
        if 'rv' in cat.columns:
            ok = cat[['obsid', 'rv']].dropna()
            rvmap = dict(zip(ok['obsid'].astype('int64').to_numpy(),
                             ok['rv'].astype(float).to_numpy()))
        say('  rv 映射: %d 条（RV 中位 %.1f km/s，|RV| 最大 %.1f km/s）'
            % (len(rvmap), float(cat['rv'].median()) if 'rv' in cat else float('nan'),
               float(cat['rv'].abs().max()) if 'rv' in cat else float('nan')))
    else:
        say('!! 没有 %s，红移改正将关闭，且跳过 join' % pkl)

    out = os.path.join(loader.INTERIM_DIR, 'spectra_features.csv')
    if os.path.exists(out) and '--reuse' in sys.argv:
        say('复用缓存 %s' % out)
        SF = pd.read_csv(out)
    else:
        SF = extract(limit_per_tar=limit, rvmap=rvmap)
        if SF is None or SF.empty:
            return 1
        SF.to_csv(out, index=False, encoding='utf-8-sig')
        say('-> %s' % out)

    say('\n谱线指数缺失率:')
    for c in BAND_COLS:
        if c in SF.columns:
            say('  %-8s %4d / %d (%.1f%%)'
                % (c, int(SF[c].isna().sum()), len(SF),
                   100.0 * SF[c].isna().sum() / len(SF)))

    # ---------------- join 目录 ----------------
    say('\n' + '-' * 84)
    say('与目录表 stellar join（按 OBSID）')
    say('-' * 84)
    if cat is None:
        say('!! 目录表未载入，跳过 join')
        with open(os.path.join(TABLES, 'spectra_report.txt'), 'w',
                  encoding='utf-8-sig') as f:
            f.write('\n'.join(LOG))
        return 0

    SF2 = SF.copy()
    SF2['obsid_num'] = pd.to_numeric(SF2['OBSID'], errors='coerce')
    j = SF2.merge(cat, left_on='obsid_num', right_on='obsid', how='inner')
    say('  光谱 %d 条，成功匹配目录 %d 条 (%.1f%%)'
        % (len(SF2), len(j), 100.0 * len(j) / max(len(SF2), 1)))
    if len(j) == 0:
        say('  !! 零匹配。可能日包年份与已抽取的 obsid 区间不重叠。')
        j.to_csv(os.path.join(TABLES, 'spectra_join.csv'),
                 index=False, encoding='utf-8-sig')
        with open(os.path.join(TABLES, 'spectra_report.txt'), 'w',
                  encoding='utf-8-sig') as f:
            f.write('\n'.join(LOG))
        return 0

    j.to_csv(os.path.join(TABLES, 'spectra_join.csv'), index=False,
             encoding='utf-8-sig')

    # ---------------- 相关性 ----------------
    say('\n' + '-' * 84)
    say('谱线指数 × 官方参数 的相关系数（Spearman，对非线性和离群更稳健）')
    say('-' * 84)
    rows = []
    for b in BAND_COLS:
        if b not in j.columns:
            continue
        for p in ['teff', 'logg', 'feh']:
            sub = j[[b, p]].dropna()
            if len(sub) < 50:
                continue
            r = float(sub[b].corr(sub[p], method='spearman'))
            rows.append({'band': b, 'param': p, 'n': len(sub), 'spearman': r})
    COR = pd.DataFrame(rows)
    if len(COR):
        piv = COR.pivot_table(index='band', columns='param', values='spearman')
        say(piv.to_string(float_format=lambda x: '%7.3f' % x))
        say('\n物理预期（对照）：')
        say('  · Ca II K 与 Teff 应显著负相关（K 线随温度降低而变强）')
        say('  · Mg b / Na D 与 logg 应相关（表面重力影响线宽）')
        say('  · Ca II K 与 [Fe/H] 应正相关（金属丰度越高 K 线越深）')
        COR.to_csv(os.path.join(TABLES, 'spectra_correlations.csv'),
                   index=False, encoding='utf-8-sig')

        say('\n最强相关（|ρ| 降序前 10）:')
        top = COR.reindex(COR['spearman'].abs().sort_values(ascending=False).index)
        for _, r in top.head(10).iterrows():
            say('  %-8s × %-6s ρ=%+.3f  (n=%d)'
                % (r['band'], r['param'], r['spearman'], r['n']))

    say('\n总耗时 %.1f 分' % ((time.time() - t0) / 60.0))
    with open(os.path.join(TABLES, 'spectra_report.txt'), 'w',
              encoding='utf-8-sig') as f:
        f.write('\n'.join(LOG))
    return 0


if __name__ == '__main__':
    sys.exit(main())
