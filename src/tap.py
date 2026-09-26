# -*- coding: utf-8 -*-
"""
LAMOST TAP 客户端

实测结论（2026-09-26，LAMOST DR10 v2.0）：
  * 端点 https://www.lamost.org/dr10/v2.0/voservice/tap ，公开、免登录
  * 只输出 VOTable（FORMAT=csv/json 被忽略），必须自己解析 XML
  * 不支持子查询（返回 500 Internal Server Error）
  * 不支持 ADQL 几何函数 spos()/scircle()
  * `top N` 不带 `order by` 时返回顺序不确定 → 必须显式排序才可复现
  * obsid 唯一（count(distinct obsid) == count(*)），可作游标分页键
  * 深分页耗时恒定（~18.5s / 20000 行），不受游标位置影响
"""
from __future__ import annotations

import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

TAP_ENDPOINT = 'https://www.lamost.org/dr10/v2.0/voservice/tap'
USER_AGENT = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
              '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

# 项目实际使用的字段（26 列）
# planid / spid / fiberid 是连接 FITS 光谱文件的关键：
#   日包路径形如 20111024/F5902/spec-55859-F5902_sp01-050.fits.gz
#   对应 (planid=F5902, spid=1, fiberid=50)
STELLAR_COLS = (
    'obsid, gp_id, designation, planid, spid, fiberid, mjd, ra, dec, '
    'snrg, snrr, subclass, mag_ps_g, mag_ps_r, mag_ps_i, mag_ps_z, mag_ps_y, '
    'gaia_g_mean_mag, teff, teff_err, logg, logg_err, feh, feh_err, alpha_m, rv'
)


class TapError(RuntimeError):
    """TAP 返回非 OK 状态或不可解析的响应。"""


def _tagname(el):
    return el.tag.split('}')[-1]


def parse_votable(blob):
    """解析 VOTable XML → (status, message, fields, rows)。"""
    text = blob.decode('utf-8', 'replace') if isinstance(blob, bytes) else blob
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise TapError('响应不是合法 VOTable：%s | 头部=%r'
                       % (exc, text[:180])) from exc

    status, message = None, ''
    for el in root.iter():
        if _tagname(el) == 'INFO' and el.get('name') == 'QUERY_STATUS':
            status = el.get('value')
            message = (el.text or '').strip()

    fields, rows = [], []
    for table in root.iter():
        if _tagname(table) != 'TABLE':
            continue
        for child in table:
            if _tagname(child) == 'FIELD':
                fields.append(child.get('name'))
        for tr in table.iter():
            if _tagname(tr) == 'TR':
                rows.append([(td.text or '') for td in tr if _tagname(td) == 'TD'])
        break
    return status, message, fields, rows


def query(adql, timeout=900, attempts=4, retry_wait=3.0):
    """执行一条 ADQL，返回 (fields, rows)。失败自动重试。"""
    url = TAP_ENDPOINT + '/sync?' + urllib.parse.urlencode(
        {'REQUEST': 'doQuery', 'LANG': 'ADQL', 'QUERY': adql, 'FORMAT': 'votable'})
    last = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                blob = resp.read()
            status, message, fields, rows = parse_votable(blob)
            if status != 'OK':
                raise TapError('QUERY_STATUS=%s %s' % (status, message[:200]))
            return fields, rows
        except Exception as exc:  # noqa: BLE001 - 重试后统一抛出
            last = exc
            if i < attempts - 1:
                time.sleep(retry_wait * (i + 1))
    raise TapError('查询重试 %d 次仍失败：%s' % (attempts, last))


def scalar(adql, timeout=900):
    """执行只返回一个数的 ADQL，直接返回该数（int）。"""
    _, rows = query(adql, timeout=timeout)
    if not rows or not rows[0]:
        return None
    try:
        return int(float(rows[0][0]))
    except (TypeError, ValueError):
        return None


def count(where='obsid > 0', timeout=900):
    return scalar('select count(*) as n from stellar where %s' % where, timeout=timeout)


def distinct(column, where='obsid > 0', timeout=900):
    return scalar('select count(distinct %s) as n from stellar where %s' % (column, where),
                  timeout=timeout)


def cursor_page(cursor, page_size=20000, cols=STELLAR_COLS, timeout=900):
    """按 obsid 游标取下一页，返回 (fields, rows)。"""
    adql = ('select top %d %s from stellar where obsid > %d order by obsid'
            % (page_size, cols, cursor))
    return query(adql, timeout=timeout, attempts=4)


def single_star(designation, cols=STELLAR_COLS, timeout=600):
    """取单颗星的全部观测（按 mjd 排序）。"""
    adql = ("select %s from stellar where designation = '%s' order by mjd"
            % (cols, designation))
    return query(adql, timeout=timeout)


if __name__ == '__main__':
    f, r = query('select top 3 obsid, teff, logg, feh from stellar order by obsid')
    print('字段:', f)
    for row in r:
        print(' ', row)
