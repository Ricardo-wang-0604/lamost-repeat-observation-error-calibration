# -*- coding: utf-8 -*-
"""
LAMOST FITS 光谱读取与谱线指数

日包路径规则
------------
    20111024/F5902/spec-55859-F5902_sp01-050.fits.gz
                      ^mjd  ^planid    ^spid ^fiberid

FITS 结构（实测）
-----------------
    HDU0 PRIMARY : OBSID, PLANID, SPID, FIBERID, MJD, DATE-OBS, RA, DEC,
                   SNRG, SNRR, CLASS, SUBCLASS（仅表头，无数据）
    HDU1 COADD   : FLUX, IVAR, WAVELENGTH, ANDMASK, ORMASK, NORMALIZATION

关键点
------
* HDU0 头里的 OBSID 与目录表 stellar.obsid 是同一个编号 → 可直接 join
* IVAR 提供逐像元逆方差 → 可做误差加权的谱线测量
* ANDMASK / ORMASK 用于剔除坏像元与天空线污染

谱线指数
--------
采用经典 Lick 系统风格的**窄带指数**：在谱线中心带测等值宽度，
两侧连续谱带取均值做内插连续谱。

    EW = ∫ (1 - F(λ)/F_c(λ)) dλ   （单位 Å，负值表示发射）

带定义（静止波长，Å），并按 LAMOST 实际波长范围裁剪：
    Ca II K  3933.7  蓝 3900-3920  红 4000-4020
    Hδ       4101.7  蓝 4080-4100  红 4120-4140
    G 带     4300    蓝 4260-4285  红 4315-4340
    Hγ       4340.5  蓝 4320-4335  红 4360-4380
    Hβ       4861.3  蓝 4840-4855  红 4870-4890
    Mg b     5175    蓝 5140-5160  红 5190-5210
    Na D     5893    蓝 5865-5885  红 5905-5925
    Hα       6562.8  蓝 6540-6555  红 6570-6590

注意：这些是**观测波长**。太阳型恒星的视向速度会带来 ~几十 km/s 的位移
（约 0.1-0.3 Å），对宽窄带指数影响可忽略；若做精细丰度分析需先退红移。
"""
from __future__ import annotations

import os
import re
import tarfile

import numpy as np

# ---------------------------------------------------------------- 带定义

BANDS = {
    'CaK':  {'c': 3933.7, 'b': (3900.0, 3920.0), 'r': (4000.0, 4020.0)},
    'Hd':   {'c': 4101.7, 'b': (4080.0, 4100.0), 'r': (4120.0, 4140.0)},
    'Gband': {'c': 4300.0, 'b': (4260.0, 4285.0), 'r': (4315.0, 4340.0)},
    'Hg':   {'c': 4340.5, 'b': (4320.0, 4335.0), 'r': (4360.0, 4380.0)},
    'Hb':   {'c': 4861.3, 'b': (4840.0, 4855.0), 'r': (4870.0, 4890.0)},
    'Mgb':  {'c': 5175.0, 'b': (5140.0, 5160.0), 'r': (5190.0, 5210.0)},
    'NaD':  {'c': 5893.0, 'b': (5865.0, 5885.0), 'r': (5905.0, 5925.0)},
    'Ha':   {'c': 6562.8, 'b': (6540.0, 6555.0), 'r': (6570.0, 6590.0)},
}


# ---------------------------------------------------------------- 读文件

def parse_name(member_name):
    """从日包成员名解析 (mjd, planid, spid, fiberid)。失败返回 None。"""
    base = os.path.basename(member_name)
    if base.endswith('.gz'):
        base = base[:-3]
    if base.endswith('.fits'):
        base = base[:-5]
    m = re.match(r'^spec-(\d+)-([A-Za-z0-9]+)_sp(\d+)-(\d+)$', base)
    if not m:
        return None
    return {'mjd': int(m.group(1)), 'planid': m.group(2),
            'spid': int(m.group(3)), 'fiberid': int(m.group(4))}


def read_spectrum(src, gzipped=False):
    """
    读取单个 LAMOST 光谱，返回 dict：
      header  : HDU0 头（dict）
      wave    : 波长数组 (Å)
      flux    : 流量数组
      ivar    : 逆方差数组（可能为 None）
      andmask : 掩膜数组（可能为 None）

    src 可以是文件路径，也可以是二进制文件对象。
    日包里的成员是 .fits.gz，用 tarfile.extractfile() 拿到的是**压缩流**，
    因此从 tar 里读时必须 gzipped=True。
    """
    import gzip as _gzip

    from astropy.io import fits

    fobj = src
    if gzipped and not isinstance(src, (str, bytes, os.PathLike)):
        fobj = _gzip.GzipFile(fileobj=src)

    with fits.open(fobj) as hdul:
        h = hdul[0].header
        header = {}
        for k in h:
            ks = str(k).strip()
            if not ks or ks in ('COMMENT', 'HISTORY', 'END', ''):
                continue
            try:
                header[ks] = h[k]
            except Exception:  # noqa: BLE001 - 无值卡片
                continue
        if len(hdul) < 2 or hdul[1].data is None:
            return {'header': header, 'wave': None, 'flux': None,
                    'ivar': None, 'andmask': None, 'ormask': None, 'norm': None}
        d = hdul[1].data
        names = [str(n).upper() for n in d.columns.names]

        def col(*cands):
            for c in cands:
                if c in names:
                    arr = d[c]
                    if arr.ndim > 1:
                        arr = arr[0]
                    return np.asarray(arr, dtype=np.float64)
            return None

        return {
            'header': header,
            'wave': col('WAVELENGTH', 'WAVE'),
            'flux': col('FLUX'),
            'ivar': col('IVAR'),
            'andmask': col('ANDMASK'),
            'ormask': col('ORMASK'),
            'norm': col('NORMALIZATION'),
        }


def iter_tar(tar_path, member_filter=None, limit=None, on_error=None):
    """流式遍历日包，逐个 yield 光谱 dict（已带 name / parsed 字段）。"""
    n = 0
    with tarfile.open(tar_path, 'r:gz') as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith('.fits.gz')):
                continue
            if member_filter and not member_filter(m.name):
                continue
            fh = tf.extractfile(m)
            if fh is None:
                continue
            try:
                sp = read_spectrum(fh, gzipped=True)
            except Exception as exc:  # noqa: BLE001
                if on_error is not None:
                    on_error(m.name, exc)
                continue
            sp['name'] = m.name
            sp['parsed'] = parse_name(m.name)
            yield sp
            n += 1
            if limit and n >= limit:
                return


# ---------------------------------------------------------------- 谱线指数

def _band_median(wave, flux, lo, hi, mask=None, min_pts=3):
    """窄带内 flux 的中位数（对坏像元与噪声稳健）。"""
    sel = (wave >= lo) & (wave <= hi)
    if mask is not None:
        sel = sel & (mask == 0)
    if sel.sum() < min_pts:
        return np.nan
    v = flux[sel]
    v = v[np.isfinite(v)]
    if len(v) < min_pts:
        return np.nan
    return float(np.median(v))


def band_indices(wave, flux, ivar=None, badmask=None, bands=None,
                 min_snr=3.0):
    """
    计算各谱线的等值宽度（Å）。

    EW = Σ (1 - F_i/F_c(λ_i)) · Δλ ，F_c 由两侧连续谱带**中位数**线性内插。

    为什么必须加守卫
    ----------------
    LAMOST 的 FLUX **未做连续谱归一化**，且红星的蓝端流量可以低到接近 0。
    若侧带连续谱估计 F_c ≈ 0，`1 - F/F_c` 会发散出上千埃的荒谬值。
    因此这里要求：
      1) 两侧带中位数均为正；
      2) 线心带与两侧带在波长网格上均有足够采样（避免空带）；
      3) 归一化后的连续谱信噪比不低于 min_snr。
    不满足则返回 NaN，由上层用缺失率而非伪造值表达。

    返回 dict: 谱线名 -> EW(Å)；不可靠时为 NaN。
    另加 'n_ok' 表示成功测得的谱线条数。
    """
    bands = bands or BANDS
    out = {}
    if wave is None or flux is None or len(wave) < 50:
        out.update({k: np.nan for k in bands})
        out['n_ok'] = 0
        return out

    order = np.argsort(wave)
    wave = np.asarray(wave, dtype=np.float64)[order]
    flux = np.asarray(flux, dtype=np.float64)[order]
    if badmask is not None:
        badmask = np.asarray(badmask)[order]
    dl = float(np.median(np.diff(wave))) if len(wave) > 1 else 1.0
    wmin, wmax = float(wave[0]), float(wave[-1])

    n_ok = 0
    for name, spec in bands.items():
        c = spec['c']
        blo, bhi = spec['b']
        rlo, rhi = spec['r']
        # 覆盖判据：整段（蓝带左端到红带右端）必须落在波长网格内
        if blo < wmin or rhi > wmax:
            out[name] = np.nan
            continue

        fb = _band_median(wave, flux, blo, bhi, badmask)
        fr = _band_median(wave, flux, rlo, rhi, badmask)
        if not (np.isfinite(fb) and np.isfinite(fr)) or fb <= 0 or fr <= 0:
            out[name] = np.nan
            continue

        cbm = 0.5 * (blo + bhi)
        crm = 0.5 * (rlo + rhi)
        sel = (wave >= cbm) & (wave <= crm)
        if badmask is not None:
            sel = sel & (badmask == 0)
        # 线心带必须真的存在
        line_sel = (wave >= c - 12.0) & (wave <= c + 12.0)
        if badmask is not None:
            line_sel = line_sel & (badmask == 0)
        if sel.sum() < 5 or line_sel.sum() < 3:
            out[name] = np.nan
            continue

        wv = wave[sel]
        slope = (fr - fb) / (crm - cbm) if crm != cbm else 0.0
        fc = fb + slope * (wv - cbm)
        good = np.isfinite(fc) & (fc > 0)
        if good.sum() < 5:
            out[name] = np.nan
            continue

        # 连续谱信噪比守卫：用线心附近的伪连续谱强度与噪声比较
        f_line = float(np.median(flux[line_sel]))
        noise = float(np.median(np.abs(np.diff(flux[line_sel])))) / np.sqrt(2) if line_sel.sum() > 3 else np.nan
        if np.isfinite(noise) and noise > 0:
            if abs(fb) / noise < min_snr:
                out[name] = np.nan
                continue

        ratio = flux[sel][good] / fc[good]
        # 单点偏差截断，避免单个宇宙线/坏像元主导
        ratio = np.clip(ratio, -5.0, 5.0)
        ew = float(np.sum(1.0 - ratio)) * dl
        if not np.isfinite(ew) or abs(ew) > 300.0:
            out[name] = np.nan
            continue
        out[name] = ew
        n_ok += 1

    out['n_ok'] = n_ok
    return out


def continuum_slope(wave, flux, windows=((4000, 4200), (6000, 6200))):
    """两个窗口伪连续谱的比，作为粗略的温度/色指标（越大越蓝越热）。"""
    if wave is None or flux is None:
        return np.nan
    a = _band_median(wave, flux, *windows[0])
    b = _band_median(wave, flux, *windows[1])
    if not (np.isfinite(a) and np.isfinite(b)) or b <= 0 or a <= 0:
        return np.nan
    return float(a / b)


def spectrum_features(sp):
    """从 read_spectrum 结果提取一行特征（含头信息）。"""
    h = sp.get('header', {})
    row = {
        'name': sp.get('name'),
        'OBSID': h.get('OBSID'),
        'PLANID': h.get('PLANID'),
        'SPID': h.get('SPID'),
        'FIBERID': h.get('FIBERID'),
        'MJD': h.get('MJD'),
        'RA': h.get('RA'),
        'DEC': h.get('DEC'),
        'SNRG': h.get('SNRG'),
        'SNRR': h.get('SNRR'),
        'CLASS': h.get('CLASS'),
        'SUBCLASS': h.get('SUBCLASS'),
    }
    wave = sp.get('wave')
    flux = sp.get('flux')
    if wave is not None:
        row['wave_min'] = float(np.nanmin(wave))
        row['wave_max'] = float(np.nanmax(wave))
        row['wave_n'] = int(len(wave))
    row.update(band_indices(wave, flux, sp.get('ivar'), sp.get('andmask')))
    row['contslope'] = continuum_slope(wave, flux)
    if flux is not None:
        with np.errstate(invalid='ignore'):
            row['flux_median'] = float(np.nanmedian(flux))
            row['flux_p16'] = float(np.nanpercentile(flux, 16))
            row['flux_p84'] = float(np.nanpercentile(flux, 84))
    return row


if __name__ == '__main__':
    import sys
    p = sys.argv[1]
    sp = read_spectrum(p)
    print({k: v for k, v in sp.items() if k in ('wave', 'flux')} and 'ok')
    print(spectrum_features(sp))
