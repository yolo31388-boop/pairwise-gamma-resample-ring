"""Gamma 正确重采样与振铃：修复实现。

本模块把题面里的六类缺陷收束到一条固定管线：

    解码 sRGB -> 预乘 alpha -> 线性光重采样 -> 反预乘 -> 编码 sRGB

对应的修复点：

1. ``linear_space`` / ``resize``
   插值一律在场景线性光下进行。sRGB 编码值上的加权平均等价于在有
   gamma 的空间里求平均，会抬高暗部（暗部边缘出现不自然的亮边、缩略
   图整体偏亮）。解码后再插值即可消除。

2. ``truncate_kernel`` / ``resize``
   Lanczos 核的负瓣在放大时会让高对比边缘过冲到负值/超白，形成光晕。
   重采样时把核截断到主瓣（只保留第一个零点以内的部分），权重因此
   非负，归一化后不再产生过冲。

3. ``uniform_factor`` / ``validate_scale``
   宽高比变化时若按目标尺寸独立计算横/纵缩放因子，两个方向的有效核
   不一致，非等比缩放的细节出现方向性失真。统一使用
   ``min(sx, sy)`` 作为滤波因子，两轴共享同一族核。

4. ``fixed_order`` / ``export``
   批处理中"先裁后缩"与"先缩后裁"只能有一条路径。固定为
   crop -> resize -> export，且裁剪永远发生在原始分辨率、重采样之前。

5. ``premult_alpha`` / ``resize``
   带 alpha 的图像在插值前预乘、插值后反预乘。否则半透明像素的透明
   颜色会被混进可见边缘，合成到画布上出现黑边。

6. ``kernel_width`` / ``resize``
   核的支撑随缩放比例变化：缩小时加宽（低通，防止混叠），放大时保持
   基础宽度；横纵两轴共用上面那个统一因子。

约定：像素为 float ``(H, W, C)``，C 为 1、3 或 4。3 通道视为 sRGB
编码的 RGB，4 通道把最后一通道视为线性覆盖度 alpha。
"""

from __future__ import annotations

import math
import os

import numpy as np

__all__ = [
    "linear_space",
    "truncate_kernel",
    "uniform_factor",
    "fixed_order",
    "premult_alpha",
    "load",
    "resize",
    "crop",
    "export",
    "kernel_width",
    "validate_scale",
    "metrics",
    "audit",
    "fallback",
    "determinism",
    "srgb_to_linear",
    "linear_to_srgb",
    "PIPELINE_ORDER",
]

# 固定处理顺序。任何入口只能按这个顺序办事。
PIPELINE_ORDER = ("crop", "resize", "export")

_EPS = 1e-12
# 缩放因子与 1 的差小于该值时视为恒等，跳过卷积。
_IDENTITY_EPS = 1e-9
# 反预乘时 alpha 低于该值视为全透明，避免除零放大噪声。
_ALPHA_EPS = 1e-6


# ---------------------------------------------------------------------------
# sRGB 传递函数
# ---------------------------------------------------------------------------

def srgb_to_linear(values):
    """把 sRGB 编码值解码成场景线性光（IEC 61966-2-1 逆传递函数）。

    ``values`` 可以是标量、数组或 ``load()`` 返回的字典。对 [-1, 0] 的
    负值做奇对称延拓：重采样后的过冲值不会凭空引入 DC 偏移。
    """
    if isinstance(values, dict):
        values = values["pixels"]
    arr = np.asarray(values, dtype=np.float64)
    mag = np.abs(arr)
    lin = np.where(
        mag <= 0.04045,
        mag / 12.92,
        ((mag + 0.055) / 1.055) ** 2.4,
    )
    out = np.sign(arr) * lin
    if out.ndim == 0:
        return float(out)
    return out


def linear_to_srgb(values):
    """把场景线性光编码成 sRGB（IEC 61966-2-1 正传递函数）。"""
    if isinstance(values, dict):
        values = values["pixels"]
    arr = np.asarray(values, dtype=np.float64)
    mag = np.abs(arr)
    enc = np.where(
        mag <= 0.0031308,
        12.92 * mag,
        1.055 * np.power(np.maximum(mag, 0.0), 1.0 / 2.4) - 0.055,
    )
    out = np.sign(arr) * enc
    if out.ndim == 0:
        return float(out)
    return out


def linear_space(values, inverse=False):
    """线性空间重采样所需的编解码入口。

    默认把 sRGB 编码值转到线性光；``inverse=True`` 时转回显示编码。
    这是题面第一条修复的公开入口：插值必须发生在这里的输出空间里。
    """
    if inverse:
        return linear_to_srgb(values)
    return srgb_to_linear(values)


# ---------------------------------------------------------------------------
# 核与截断
# ---------------------------------------------------------------------------

def _lanczos_raw(d, a):
    return np.sinc(d) * np.sinc(d / a)


def _catmull_raw(d):
    ax = np.abs(d)
    out = np.zeros_like(ax)
    lo = ax <= 1.0
    hi = (ax > 1.0) & (ax < 2.0)
    out[lo] = 1.5 * ax[lo] ** 3 - 2.5 * ax[lo] ** 2 + 1.0
    out[hi] = -0.5 * ax[hi] ** 3 + 2.5 * ax[hi] ** 2 - 4.0 * ax[hi] + 2.0
    return out


def _triangle_raw(d):
    return np.clip(1.0 - np.abs(d), 0.0, None)


def _box_raw(d):
    return np.where(np.abs(d) < 0.5, 1.0, 0.0)


def _kernel_spec(kernel):
    """返回 (原始核函数, 原始支撑, 主瓣截断点)。"""
    if callable(kernel):
        support = float(getattr(kernel, "support", 1.0))
        return kernel, support, support

    name = str(kernel).strip().lower()
    if name in ("lanczos", "lanczos3"):
        a = 3.0
        return (lambda d: _lanczos_raw(d, a)), a, 1.0
    if name == "lanczos2":
        a = 2.0
        return (lambda d: _lanczos_raw(d, a)), a, 1.0
    if name == "lanczos4":
        a = 4.0
        return (lambda d: _lanczos_raw(d, a)), a, 1.0
    if name in ("box", "nearest"):
        return _box_raw, 0.5, 0.5
    if name in ("triangle", "bilinear", "linear"):
        return _triangle_raw, 1.0, 1.0
    if name in ("catmull", "catmull-rom", "bicubic", "cubic"):
        return _catmull_raw, 2.0, 1.0
    raise ValueError(f"unknown kernel: {kernel!r}")


def _kernel_values(kernel, d):
    """按截断到主瓣后的核求权重；返回值非负且主瓣外为 0。"""
    raw, _support, cut = _kernel_spec(kernel)
    v = np.asarray(raw(np.asarray(d, dtype=np.float64)), dtype=np.float64)
    # 圆形截断 + 负瓣清零：这两步一起消除振铃。
    v = np.where(np.abs(d) < cut, np.maximum(v, 0.0), 0.0)
    return v


def truncate_kernel(x, a=3.0, support=None, kernel="lanczos"):
    """把核截断到主瓣，消掉 Lanczos 负瓣带来的振铃。

    参数：
        x: 采样坐标（标量或数组；单位是"核半径"归一化后的坐标）。
        a: Lanczos 参数（默认 3，即 Lanczos3）。传入核名时它被当作
           ``kernel``，便于 ``truncate_kernel(x, "lanczos2")`` 这种写法。
        support: 自定义截断半径；默认取核的第一个零点（Lanczos 为 1）。
        kernel: 核名或 callable。

    返回与 ``x`` 同形状的非负权重，``|x| >= cut`` 处为 0。截断后的
    权重在重采样时会按每个输出像素重新归一化，因此不引入亮度偏移。
    """
    if isinstance(a, str):
        # truncate_kernel(x, "lanczos2") 形式。
        kernel, a = a, 3.0
    a = float(a)

    if callable(kernel):
        raw, _support, default_cut = _kernel_spec(kernel)
    elif str(kernel).strip().lower() in ("lanczos", "lanczos3", "lanczos4"):
        raw = lambda d: _lanczos_raw(d, a)  # noqa: E731
        default_cut = 1.0
    else:
        raw, _support, default_cut = _kernel_spec(kernel)

    cut = default_cut if support is None else float(support)
    d = np.asarray(x, dtype=np.float64)
    w = np.asarray(raw(d), dtype=np.float64)
    out = np.where(np.abs(d) < cut, np.maximum(w, 0.0), 0.0)
    if out.ndim == 0:
        return float(out)
    return out


def kernel_width(kernel="lanczos", factor=1.0):
    """核的支撑宽度（以源像素为单位的全宽），按缩放比例定宽。

    * 缩小（factor < 1）时核加宽：``2 * support / factor``，
      这样才有正确的低通，缩略图不会混叠。
    * 放大（factor >= 1）时保持基础宽度 ``2 * support``。

    宽度按截断后的主瓣计算，与 ``resize`` 内部实际使用的核一致。
    """
    if not isinstance(kernel, (str,)) and not callable(kernel):
        # 兼容 kernel_width(factor) 写法。
        factor, kernel = float(kernel), "lanczos"
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("scale factor must be finite and > 0")
    factor = max(factor, 1e-9)

    _raw, _support, cut = _kernel_spec(kernel)
    scale = 1.0 / factor if factor < 1.0 else 1.0
    width = max(2.0 * cut * scale, 1.0)
    return float(width)


def _support_radius(kernel):
    """截断后核的半径（主瓣边界），供重采样确定抽头范围。"""
    return _kernel_spec(kernel)[2]


# ---------------------------------------------------------------------------
# 缩放因子与校验
# ---------------------------------------------------------------------------

def _pair(x):
    if isinstance(x, (tuple, list, np.ndarray)) and len(x) == 2:
        return float(x[0]), float(x[1])
    return None


def _ratio(src, dst):
    src, dst = float(src), float(dst)
    if not math.isfinite(src) or src <= 0.0:
        raise ValueError("source size must be finite and > 0")
    if not math.isfinite(dst) or dst <= 0.0:
        raise ValueError("target size must be finite and > 0")
    return dst / src


def uniform_factor(src_size=None, dst_size=None):
    """等比统一因子 ``min(sx, sy)``。

    支持的调用形式：
        uniform_factor((sw, sh), (dw, dh))   两轴尺寸 -> min(dw/sw, dh/sh)
        uniform_factor([sx, sy])             直接给两个因子 -> min
        uniform_factor(sw, dw)               单轴因子
        uniform_factor(factor)               原样返回（> 0）

    横竖两次缩放必须共享同一个有效核，因此非等比输出也把两个方向都
    锁到这个最小值上。
    """
    if dst_size is None:
        if src_size is None:
            return 1.0
        if np.isscalar(src_size) or isinstance(src_size, np.generic):
            f = float(src_size)
            if not math.isfinite(f) or f <= 0.0:
                raise ValueError("scale factor must be finite and > 0")
            return f
        # 单参数下两元素序列一律视为两个因子。
        vals = [float(v) for v in src_size]
        if not vals:
            return 1.0
        return min(vals)

    src_pair = _pair(src_size)
    dst_pair = _pair(dst_size)
    if src_pair is not None and dst_pair is not None:
        return min(_ratio(src_pair[0], dst_pair[0]),
                   _ratio(src_pair[1], dst_pair[1]))
    return _ratio(src_size, dst_size)


def validate_scale(src_size=None, dst_size=None, factor=None, tolerance=1e-6):
    """校验缩放因子与尺寸，返回规整后的报告；非法输入抛 ``ValueError``。

    报告字段：
        factor / sx / sy / uniform / upscale / downscale / identity
    """
    if factor is None:
        factor = uniform_factor(src_size, dst_size)
    factor = float(factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError("scale factor must be finite and > 0")

    src_pair = _pair(src_size)
    dst_pair = _pair(dst_size)
    if src_pair is not None and dst_pair is not None:
        sx = _ratio(src_pair[0], dst_pair[0])
        sy = _ratio(src_pair[1], dst_pair[1])
    elif src_pair is None and np.isscalar(src_size) and np.isscalar(dst_size):
        sx = sy = _ratio(src_size, dst_size)
    else:
        sx = sy = factor

    return {
        "factor": factor,
        "sx": sx,
        "sy": sy,
        "uniform": abs(sx - sy) <= tolerance,
        "upscale": factor > 1.0,
        "downscale": factor < 1.0,
        "identity": abs(factor - 1.0) <= _IDENTITY_EPS,
    }


# ---------------------------------------------------------------------------
# 处理顺序
# ---------------------------------------------------------------------------

def fixed_order(stages=None):
    """返回固定处理顺序 ``crop -> resize -> export``。

    ``stages`` 为 None 时返回完整顺序；给出部分阶段时按固定顺序排好并
    校验没有重复/未知阶段。非序列输入按"未指定"处理，保证入口幂等。
    """
    if stages is None:
        return list(PIPELINE_ORDER)
    if isinstance(stages, str):
        stages = [stages]
    if np.isscalar(stages) or not hasattr(stages, "__iter__"):
        # 非阶段序列的输入视为"未指定"，入口保持幂等。
        return list(PIPELINE_ORDER)

    order = []
    for st in stages:
        key = str(st).strip().lower()
        if key not in PIPELINE_ORDER:
            raise ValueError(f"unknown stage: {st!r}")
        if key in order:
            raise ValueError(f"duplicate stage: {key!r}")
        order.append(key)
    order.sort(key=PIPELINE_ORDER.index)
    return order


# ---------------------------------------------------------------------------
# 预乘 alpha
# ---------------------------------------------------------------------------

def _split_rgba(arr):
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim < 2:
        return arr, None
    if arr.shape[-1] == 4:
        return arr[..., :3], arr[..., 3:4]
    if arr.shape[-1] == 3:
        return arr, None
    if arr.shape[-1] == 2:
        return arr[..., :1], arr[..., 1:]
    return arr, None


def premult_alpha(rgb, alpha=None, mode="premultiply"):
    """预乘 / 反预乘 alpha。

    * alpha 为 None 且输入有 4 通道：在最后一通道上就地预乘/反预乘。
    * alpha 给出：只处理颜色，返回与 ``rgb`` 同形状的结果。

    预乘必须在重采样之前：否则半透明像素插值时会把透明区域的颜色
    （通常是无意义的 0）混进可见边缘，合成后就是黑边。
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    a_given = alpha is not None
    if alpha is None:
        rgb, a = _split_rgba(rgb)
        if a is None:
            out = rgb
            return float(out) if out.ndim == 0 else out
    else:
        a = np.asarray(alpha, dtype=np.float64)
        if a.ndim == rgb.ndim - 1:
            a = a[..., None]

    if rgb.shape[:2] != a.shape[:2] and rgb.ndim >= 2:
        raise ValueError("alpha and image must share spatial size")

    m = str(mode).strip().lower()
    legacy_alpha = not a_given and alpha is None
    if m in ("premultiply", "premult", "pre", "multiply"):
        result = rgb * a
    elif m in ("unpremultiply", "unpremult", "unpremultiplied", "divide"):
        safe = np.where(a > _ALPHA_EPS, a, 1.0)
        result = np.where(a > _ALPHA_EPS, rgb / safe, 0.0)
    else:
        raise ValueError(f"unknown alpha mode: {mode!r}")

    if legacy_alpha:
        result = np.concatenate([result, a], axis=-1)
    if result.ndim == 0:
        return float(result)
    return result


# ---------------------------------------------------------------------------
# 图像读取 / 归一化
# ---------------------------------------------------------------------------

def _normalize_pixels(data):
    """把各种输入规整成 float 图像 ``(H, W, C)``，C 为 3(RGB) 或 4(RGBA)。"""
    normalized = False
    if isinstance(data, dict):
        normalized = bool(data.get("normalized", False))
        data = data.get("pixels", data)
    arr = np.asarray(data)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1, 1)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1, 1)
    if arr.ndim == 2:
        arr = arr[..., None]
    if arr.ndim > 3:
        arr = arr.reshape(arr.shape[0], arr.shape[1], -1)

    if np.issubdtype(arr.dtype, np.integer):
        maxv = 65535.0 if arr.dtype.itemsize > 1 else 255.0
        arr = arr.astype(np.float64) / maxv
    else:
        arr = arr.astype(np.float64)
        if arr.size and not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
        # 裸 float 数组按常见约定推断量纲；经 load() 规整过的图像跳过这一步，
        # 以免把线性光/浮点 HDR 数据误当作 0..255。
        if not normalized and arr.size and float(np.max(arr)) > 1.5:
            arr = arr / 255.0

    # 统一到 RGB / RGBA：灰度复制成 RGB，灰度+alpha 展开成 RGBA。
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    elif arr.shape[-1] == 2:
        arr = np.concatenate([np.repeat(arr[..., :1], 3, axis=-1), arr[..., 1:]],
                             axis=-1)
    return arr


def load(source, colorspace="srgb"):
    """读取图像，返回 ``{"pixels", "srgb", "alpha", "size"}`` 字典。

    ``source`` 可以是路径（经 Pillow 读入）、数组、嵌套列表或标量。
    读入后像素为 sRGB 编码值；``colorspace="linear"`` 表示输入已经在线性光下。
    """
    if isinstance(source, (str, os.PathLike)) or hasattr(source, "__fspath__"):
        from PIL import Image

        with Image.open(source) as im:
            has_alpha = im.mode in ("RGBA", "LA", "PA")
            im = im.convert("RGBA" if has_alpha else "RGB")
            arr = np.asarray(im, dtype=np.float64) / 255.0
    else:
        arr = _normalize_pixels(source)

    if arr.shape[-1] not in (1, 3, 4):
        raise ValueError("only 1/3/4 channel images are supported")

    cs = str(colorspace).strip().lower()
    if cs not in ("srgb", "gamma", "encoded", "linear", "lin", "scene"):
        raise ValueError(f"unknown colorspace: {colorspace!r}")
    is_srgb = cs not in ("linear", "lin", "scene")

    return {
        "pixels": arr.astype(np.float64),
        "srgb": is_srgb,
        "alpha": arr.shape[-1] == 4,
        "size": (int(arr.shape[1]), int(arr.shape[0])),
        "normalized": True,
    }


# ---------------------------------------------------------------------------
# 重采样核心
# ---------------------------------------------------------------------------

def _resample_axis(arr, axis, dst_len, ratio, filter_scale, kernel):
    """沿单轴做可分离重采样。

    ``ratio``         该轴的目标/源像素比（决定采样点位置）。
    ``filter_scale``  核宽因子（等比统一因子或该轴缩放因子，决定低通强度）。
    """
    n = arr.shape[axis]
    dst_len = int(dst_len)
    if dst_len <= 0:
        raise ValueError("target length must be > 0")
    fs = float(filter_scale)
    if dst_len == n:
        # 该轴目标长度不变：像素网格不动，不做任何卷积。这样等比统一因子
        # 不会把未缩放的轴一起糊掉。
        return np.asarray(arr, dtype=np.float64)

    cut = _support_radius(kernel)
    scale = min(fs, 1.0)              # 只有缩小才加宽核
    half = cut / scale

    # 像素中心约定：目标像素中心映射回源坐标。
    centers = (np.arange(dst_len, dtype=np.float64) + 0.5) / ratio - 0.5

    # 每个输出像素的局部抽头窗口。窗口宽度固定，不随目标像素移动，因此
    # 权重矩阵是 (dst, taps) 而不是 (dst, n)，放大时也不会爆内存/时间。
    taps = int(math.ceil(2.0 * half)) + 1
    base = np.round(centers - half).astype(np.int64)     # 窗口左端（源像素）
    js = base[:, None] + np.arange(taps, dtype=np.int64)[None, :]

    d = (centers[:, None] - js) * scale
    w = _kernel_values(kernel, d)
    # 截断核：主瓣外已经为 0；显式再乘一次掩码防止浮点边界。
    w = np.where(np.abs(d) < cut, w, 0.0)

    # 边界用复刻（replicate）扩展：把越界下标夹回图像内，权重照常累加。
    jj = np.clip(js, 0, n - 1)

    total = w.sum(axis=1, keepdims=True)
    w = np.divide(w, total, out=np.zeros_like(w), where=total > _EPS)
    # 极罕见的总权重为 0 时退化为最近邻，保证输出仍然有效。
    zero_rows = total[:, 0] <= _EPS
    if np.any(zero_rows):
        nearest = np.clip(np.rint(centers[zero_rows]).astype(np.int64), 0, n - 1)
        w[zero_rows, :] = 0.0
        pos = np.clip(nearest - base[zero_rows], 0, taps - 1)
        w[zero_rows, pos] = 1.0

    moved = np.moveaxis(np.asarray(arr, dtype=np.float64), axis, -1)
    flat = moved.reshape(-1, n)                         # (M, n)
    w = w.astype(np.float64)
    jj = jj.astype(np.int64)

    # 分块累加：放大时 dst 很大，一次性 gather 出的 (M, dst, taps) 会占用
    # 数百 MB。按输出像素分块，把峰值内存压在 ~64MB 以内，结果不受影响
    # （每块独立求和，顺序固定，仍然逐位确定）。
    out = np.empty((flat.shape[0], dst_len), dtype=np.float64)
    max_elems = 8_000_000
    step = max(1, max_elems // max(1, flat.shape[0] * taps))
    for s in range(0, dst_len, step):
        e = min(dst_len, s + step)
        gathered = flat[:, jj[s:e]]                     # (M, blk, taps)
        out[:, s:e] = np.einsum("mdt,dt->md", gathered, w[s:e], optimize=True)

    out = out.reshape(moved.shape[:-1] + (dst_len,))
    return np.moveaxis(out, -1, axis)


def _decode_image(px, srgb):
    """sRGB 解码，alpha 通道不参与 gamma 变换。"""
    if not srgb:
        return np.asarray(px, dtype=np.float64)
    rgb, a = _split_rgba(px)
    lin = srgb_to_linear(rgb)
    if a is None:
        return lin
    return np.concatenate([lin, a], axis=-1)


def _encode_image(px, srgb):
    """线性光编码回 sRGB，alpha 通道原样保留。"""
    if not srgb:
        return np.asarray(px, dtype=np.float32)
    rgb, a = _split_rgba(px)
    enc = linear_to_srgb(np.clip(rgb, 0.0, 1.0))
    if a is None:
        return enc.astype(np.float32)
    return np.concatenate([np.clip(enc, 0.0, 1.0), np.clip(a, 0.0, 1.0)],
                          axis=-1).astype(np.float32)


def _premultiply_image(lin):
    """在线性光下预乘 alpha；返回预乘后的工作图像。"""
    rgb, a = _split_rgba(lin)
    if a is None:
        return lin
    return np.concatenate([rgb * a, a], axis=-1)


def _unpremultiply_image(work):
    rgb, a = _split_rgba(work)
    if a is None:
        return work
    safe = np.where(a > _ALPHA_EPS, a, 1.0)
    rgb_out = np.where(a > _ALPHA_EPS, rgb / safe, 0.0)
    return np.concatenate([rgb_out, np.clip(a, 0.0, 1.0)], axis=-1)


def _target_size(size, shape, factor):
    h, w = shape[0], shape[1]
    if size is not None:
        pair = _pair(size)
        if pair is None:
            raise ValueError("size must be (width, height)")
        dw, dh = int(pair[0]), int(pair[1])
        if dw <= 0 or dh <= 0:
            raise ValueError("target size must be > 0")
        return dw, dh
    if factor is not None:
        f = float(factor)
        if not math.isfinite(f) or f <= 0.0:
            raise ValueError("scale factor must be finite and > 0")
        return max(1, int(round(w * f))), max(1, int(round(h * f)))
    return w, h


def resize(img, size=None, factor=None, kernel="lanczos", uniform=True,
           colorspace=None, as_array=False):
    """在线性光下重采样，返回结果字典。

    参数：
        img: 图像数组或 ``load()`` 的字典（自动识别 alpha）。
        size: 目标尺寸 ``(width, height)``。
        factor: 或者直接给缩放因子（两轴同因子）。
        kernel: ``"lanczos"``（默认，截断到主瓣）、``"lanczos2"``、
            ``"catmull"``、``"triangle"``、``"box"`` 或 callable。
        uniform: True 时两轴共享 ``min(sx, sy)`` 作为滤波因子，
            消除非等比缩放的方向性失真。

    ``size`` 与 ``factor`` 都不给时原样返回（幂等），方便入口复用。
    """
    src = img
    if isinstance(src, dict):
        px = _normalize_pixels(src)
        srgb = bool(src.get("srgb", True))
    else:
        px = _normalize_pixels(src)
        srgb = True
        if colorspace is not None:
            srgb = str(colorspace).strip().lower() not in ("linear", "lin", "scene")

    h, w = px.shape[:2]
    has_alpha = px.shape[-1] == 4

    dw, dh = _target_size(size, px.shape, factor)
    if dw == w and dh == h and factor is None:
        out = px.astype(np.float32)
        meta = {
            "size": (dw, dh),
            "factor": 1.0,
            "axes": (1.0, 1.0),
            "kernel": kernel,
            "kernel_width": kernel_width(kernel, 1.0),
            "alpha": has_alpha,
            "srgb": srgb,
            "uniform": True,
            "order": list(PIPELINE_ORDER),
        }
        result = {"pixels": out, "meta": meta, "size": (dw, dh),
              "normalized": True}
        return out if as_array else result

    rx, ry = _ratio(w, dw), _ratio(h, dh)
    # 统一因子：滤波强度按两轴最小者，而采样位置仍按各自目标尺寸，
    # 于是横竖两轴使用同一族有效核，非等比缩放的细节不会出现方向性
    # 失真。uniform=False 时退回按轴独立计算（不推荐，仅供对照）。
    uni = min(rx, ry)
    fs_x = uni if uniform else rx
    fs_y = uni if uniform else ry

    # 1) 解码到线性光。
    lin = _decode_image(px, srgb)

    # 2) 预乘 alpha（若有），在工作空间里插值。
    if has_alpha:
        work = _premultiply_image(lin)
    else:
        work = lin

    # 3) 可分离重采样：先水平后垂直，顺序固定以保确定性。
    work = _resample_axis(work, 1, dw, rx, fs_x, kernel)
    work = _resample_axis(work, 0, dh, ry, fs_y, kernel)

    # 4) 反预乘 + 编码。
    if has_alpha:
        work = _unpremultiply_image(work)
    out = _encode_image(work, srgb)

    meta = {
        "size": (dw, dh),
        "factor": uni,
        "axes": (rx, ry),
        "kernel": str(kernel) if isinstance(kernel, str) else "custom",
        "kernel_width": kernel_width(kernel, uni),
        "alpha": has_alpha,
        "srgb": srgb,
        "uniform": bool(uniform),
        "order": list(PIPELINE_ORDER),
    }
    result = {"pixels": out, "meta": meta, "size": (dw, dh),
              "normalized": True}
    return out if as_array else result


# ---------------------------------------------------------------------------
# 裁剪 / 导出
# ---------------------------------------------------------------------------

def crop(img, box=None, size=None, as_array=False):
    """在原始分辨率上裁剪（固定顺序的第一步）。

    ``box`` 为 ``(left, upper, right, lower)``，单位是像素边界；None 时
    返回整幅图。``size`` 给出目标输出尺寸时，裁剪后立即重采样（仍保持
    "crop 在 resize 之前"）。
    """
    src = img
    if isinstance(src, dict):
        px = _normalize_pixels(src)
        srgb = bool(src.get("srgb", True))
    else:
        px = _normalize_pixels(src)
        srgb = True

    h, w = px.shape[:2]
    if box is None:
        x0, y0, x1, y1 = 0, 0, w, h
    else:
        pair = None
        if isinstance(box, dict):
            if "box" in box:
                pair = _pair(box["box"])
        if pair is None:
            if len(box) != 4:
                raise ValueError("box must have 4 values")
            pair = tuple(float(v) for v in box)
        x0, y0, x1, y1 = pair

    ix0 = max(0, min(int(round(x0)), w - 1))
    iy0 = max(0, min(int(round(y0)), h - 1))
    ix1 = max(ix0 + 1, min(int(round(x1)), w))
    iy1 = max(iy0 + 1, min(int(round(y1)), h))
    out = px[iy0:iy1, ix0:ix1, :].copy()

    if size is not None:
        dw, dh = _target_size(size, out.shape, None)
        if (out.shape[1], out.shape[0]) != (dw, dh):
            out = resize({"pixels": out, "srgb": srgb, "normalized": True}, size=(dw, dh),
                         as_array=True)

    meta = {"box": (ix0, iy0, ix1, iy1), "order": ["crop"]}
    meta["size"] = (int(out.shape[1]), int(out.shape[0]))
    # 保持输入的浮点精度：裁剪只是取子窗口，不做数值变换。提前降精度
    # 会让"先裁后缩"与"先缩后裁"在最后一位上分叉。
    if not np.issubdtype(out.dtype, np.floating):
        out = out.astype(np.float64)
    result = {"pixels": out, "meta": meta, "size": meta["size"],
              "normalized": True}
    return result["pixels"] if as_array else result


def export(img, path=None, size=None, box=None, kernel="lanczos",
           order=None, as_array=False):
    """按固定顺序导出：crop -> resize -> export。

    批处理里同一张图"先裁后缩"和"先缩后裁"必须只有一条路径；这里
    统一在原始分辨率裁剪，再在线性光下重采样，最后编码/落盘。
    ``order`` 若显式给出，必须与 ``PIPELINE_ORDER`` 一致。
    """
    if order is not None:
        ordered = fixed_order(order)
        if tuple(ordered) != PIPELINE_ORDER:
            raise ValueError(
                f"processing order must be {list(PIPELINE_ORDER)}, "
                f"got {ordered}"
            )

    src = img
    if isinstance(src, dict):
        px = _normalize_pixels(src)
        srgb = bool(src.get("srgb", True))
    else:
        px = _normalize_pixels(src)
        srgb = True

    src_size = (int(px.shape[1]), int(px.shape[0]))

    # 1) crop —— 永远先在原始分辨率上做。
    if box is not None:
        cropped = crop({"pixels": px, "srgb": srgb, "normalized": True},
                       box=box, as_array=True)
    else:
        cropped = px

    # 2) resize —— 在线性光、截断核、等比统一因子、预乘 alpha。
    if size is not None:
        resized = resize({"pixels": cropped, "srgb": srgb, "normalized": True},
                         size=size, kernel=kernel, as_array=True)
    else:
        # 未缩放时不额外降精度；编码保持不变。
        resized = np.ascontiguousarray(cropped)

    # 3) export —— 编码已在 resize 内完成，这里只负责写出。
    if path is not None:
        from PIL import Image

        pxc = np.clip(resized, 0.0, 1.0)
        arr8 = (pxc * 255.0 + 0.5).astype(np.uint8)
        mode = "RGBA" if arr8.shape[-1] == 4 else "RGB"
        if arr8.shape[-1] not in (3, 4):
            arr8 = np.repeat(arr8[..., :1], 3, axis=-1)
            mode = "RGB"
        Image.fromarray(arr8, mode).save(path)

    out_size = (int(resized.shape[1]), int(resized.shape[0]))
    meta = {
        "order": list(PIPELINE_ORDER),
        "src_size": src_size,
        "size": out_size,
        "kernel": kernel,
        "alpha": resized.shape[-1] == 4,
        "path": str(path) if path is not None else None,
    }
    result = {"pixels": resized, "meta": meta, "size": out_size,
              "normalized": True}
    return result["pixels"] if as_array else result


# ---------------------------------------------------------------------------
# 指标与审计
# ---------------------------------------------------------------------------

def _as_metrics_image(img, colorspace="srgb"):
    if isinstance(img, dict):
        px = _normalize_pixels(img)
        srgb = bool(img.get("srgb", True))
    else:
        px = _normalize_pixels(img)
        srgb = str(colorspace).strip().lower() not in ("linear", "lin", "scene")
    return px, srgb


def metrics(img, reference=None, colorspace="srgb"):
    """在线性光下计算质量指标。

    返回 mean / std / min / max / contrast / dark_mean / mean_neg；
    给 ``reference`` 时追加 rmse / psnr / max_abs_error。
    """
    px, srgb = _as_metrics_image(img, colorspace)
    lin = srgb_to_linear(px) if srgb else px.astype(np.float64)
    rgb = lin[..., :3] if lin.shape[-1] >= 3 else lin

    lum = (0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]) \
        if rgb.shape[-1] >= 3 else rgb[..., 0]

    report = {
        "mean": float(np.mean(rgb)),
        "std": float(np.std(rgb)),
        "min": float(np.min(rgb)),
        "max": float(np.max(rgb)),
        "contrast": float(np.max(lum) - np.min(lum)),
        "dark_mean": float(np.mean(rgb[rgb < 0.05])) if np.any(rgb < 0.05) else 0.0,
        "mean_neg": float(np.mean(rgb < 0.0)),
        "size": (int(rgb.shape[1]), int(rgb.shape[0])),
    }
    if reference is not None:
        ref_px, _ = _as_metrics_image(reference, colorspace)
        ref = srgb_to_linear(ref_px) if srgb else ref_px.astype(np.float64)
        if ref.shape != lin.shape:
            raise ValueError("reference shape does not match image shape")
        diff = lin - ref
        mse = float(np.mean(diff ** 2))
        report["rmse"] = math.sqrt(mse)
        report["psnr"] = float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)
        report["max_abs_error"] = float(np.max(np.abs(diff)))
    return report


def audit(img, stage=None, meta=None, premultiplied=None, reference=None,
          dark_tolerance=0.02):
    """审计图像是否处于正确状态，返回问题列表。

    检测项：
        * ``dark_lifted``  —— 暗部被 gamma 错误插值抬高（亮边）
        * ``ring``         —— 存在过冲/欠冲（振铃）
        * ``alpha_not_premultiplied`` —— 全透明像素仍带颜色（黑边风险）
        * ``bad_order``    —— meta 中的处理顺序不合规

    ``dark_lifted`` 需要 ``reference``（在正确空间里算出的期望图）：单张
    显示空间的图无法区分"本来就亮"与"被 gamma 插值抬亮"，所以只有在
    给出参照时才做比较，避免误报。

    ``premultiplied`` 显式声明像素是否已预乘；None 时按内容推断——
    合法管线的输出是反预乘后的显示值，因此只能靠"alpha≈0 却保留
    颜色"这种未预乘的特征来判定。
    """
    px, srgb = _as_metrics_image(img)
    if meta is None and isinstance(img, dict):
        meta = img.get("meta")

    lin = srgb_to_linear(px) if srgb else px.astype(np.float64)
    rgb = px[..., :3] if px.shape[-1] >= 3 else px

    issues = []

    # 振铃：显示值越界（>1 或 <0）。
    lo, hi = float(np.min(rgb)), float(np.max(rgb))
    if hi > 1.0 + 1e-4 or lo < -1e-4:
        issues.append("ring")

    # 暗部抬亮：拿正确结果作参照，比较共同的暗部区域在线性光下的亮度。
    # 在 sRGB 编码值上插值只会把暗部抬高，不会压低。
    if reference is not None:
        ref_px, ref_srgb = _as_metrics_image(reference)
        if ref_px.shape != px.shape:
            raise ValueError("reference shape does not match image shape")
        ref_lin = srgb_to_linear(ref_px) if ref_srgb else ref_px.astype(np.float64)
        ref_lum = ref_lin[..., :3]
        dark = np.max(ref_lum, axis=-1) < 0.05
        if np.any(dark):
            got = float(np.mean(lin[..., :3][dark]))
            want = float(np.mean(ref_lum[dark]))
            if got - want > dark_tolerance:
                issues.append("dark_lifted")

    # alpha 预乘检查。未预乘的标志是：alpha 几乎为 0 的像素仍保留着
    # 颜色。已预乘的像素在 alpha→0 时颜色也必然→0。
    if px.shape[-1] == 4 and premultiplied is not True:
        a = px[..., 3]
        invisible = a < 1e-3
        if np.any(invisible):
            color = np.max(rgb[invisible], axis=-1)
            if float(np.max(color)) > 1e-3:
                issues.append("alpha_not_premultiplied")

    order_info = None
    if meta is not None and meta.get("order"):
        order_info = list(meta["order"])
        # 任何合法管线都只能是固定顺序的前缀（crop / crop+resize / 全流程）。
        if tuple(order_info) != PIPELINE_ORDER[:len(order_info)]:
            issues.append("bad_order")

    if stage is not None:
        st = str(stage).strip().lower()
        if st not in PIPELINE_ORDER:
            raise ValueError(f"unknown stage: {stage!r}")
    else:
        st = "unknown"

    report = {
        "stage": st,
        "ok": not issues,
        "issues": issues,
        "range": (lo, hi),
        "alpha": px.shape[-1] == 4,
        "srgb": bool(srgb),
        "order": order_info,
        "size": (int(px.shape[1]), int(px.shape[0])),
    }
    return report

# ---------------------------------------------------------------------------
# 退化与确定性
# ---------------------------------------------------------------------------

def _nearest_resize(px, dw, dh):
    px = np.asarray(px, dtype=np.float32)
    h, w = px.shape[:2]
    yi = np.minimum((np.arange(dh, dtype=np.float64) + 0.5) * h / max(dh, 1),
                    h - 1).astype(np.int64)
    xi = np.minimum((np.arange(dw, dtype=np.float64) + 0.5) * w / max(dw, 1),
                    w - 1).astype(np.int64)
    return px[yi][:, xi]


def fallback(img, size=None, box=None, kernel="lanczos", **kwargs):
    """保守退化路径：任何异常输入都不崩，返回可用的结果。

    退化阶梯：请求的核 -> 截断 catmull -> 三角形 -> 最近邻。未知核名、
    非法尺寸、非有限数值都会被规整到合法值；最终仍失败时按最近邻输出，
    保证尺寸正确。
    """
    px = _normalize_pixels(img)
    srgb = bool(img.get("srgb", True)) if isinstance(img, dict) else True
    h, w = px.shape[:2]

    dw, dh = w, h
    if size is not None:
        pair = _pair(size)
        if pair is None:
            dw, dh = w, h
        else:
            dw = int(pair[0]) if math.isfinite(float(pair[0])) else w
            dh = int(pair[1]) if math.isfinite(float(pair[1])) else h
        dw, dh = max(1, dw), max(1, dh)

    requested = kernel
    degraded = False
    try:
        _kernel_spec(kernel)
    except ValueError:
        kernel = "catmull"
        degraded = True

    # 裁剪仍按固定顺序先做：裁剪框非法时退回整幅图。
    work = px
    if box is not None:
        try:
            box_pair = _pair(box)
        except (TypeError, ValueError):
            box_pair = None
        if box_pair is not None:
            try:
                work = crop({"pixels": px, "srgb": srgb, "normalized": True}, box=box_pair,
                            as_array=True)
            except ValueError:
                work = px

    for k in (kernel, "catmull", "triangle", "box"):
        try:
            res = resize({"pixels": work, "srgb": srgb, "normalized": True}, size=(dw, dh),
                         kernel=k, as_array=True)
            if res is not None and res.shape[0] == dh and res.shape[1] == dw:
                meta = {"size": (dw, dh), "kernel": k, "requested": requested,
                        "fallback": degraded or k != kernel}
                return {"pixels": res, "meta": meta, "size": (dw, dh)}
        except Exception:  # noqa: BLE001 - 退化路径必须吞掉任何失败
            degraded = True
            continue

    out = _nearest_resize(work, dw, dh)
    return {"pixels": out,
            "meta": {"size": (dw, dh), "kernel": "nearest",
                     "requested": requested, "fallback": True},
            "size": (dw, dh)}


def _determinism_probe():
    """默认探针：一张固定的合成图走完整管线。"""
    h, w = 16, 16
    yy, xx = np.mgrid[0:h, 0:w]
    base = ((xx + yy) % 4) / 3.0
    rgba = np.stack([base, 1.0 - base, 0.5 * base + 0.25,
                     0.5 + 0.5 * base], axis=-1)
    return resize({"pixels": rgba, "srgb": True, "normalized": True}, size=(8, 8), as_array=True)


def determinism(fn=None, *args, repeats=3, **kwargs):
    """验证同一输入重复执行逐位一致。

    实现中固定了轴的遍历顺序、einsum 的归约轴以及权重归一化顺序，
    不依赖随机数或全局状态，因此结果必须是位级确定的。
    ``fn`` 不可调用时用内置的固定探针走完整管线。
    """
    import hashlib

    if fn is None or not callable(fn):
        fn = _determinism_probe
        args = ()
        kwargs = {}

    hashes = []
    shapes = []
    first = None
    for _ in range(max(1, int(repeats))):
        out = fn(*args, **kwargs)
        if isinstance(out, dict):
            out = out.get("pixels", out)
        arr = np.ascontiguousarray(np.asarray(out, dtype=np.float64))
        hashes.append(hashlib.sha256(arr.tobytes()).hexdigest())
        shapes.append(tuple(arr.shape))
        if first is None:
            first = arr

    ok = len(set(hashes)) == 1 and len(set(shapes)) == 1
    return {
        "deterministic": bool(ok),
        "equal": bool(ok),
        "repeats": len(hashes),
        "hashes": hashes,
        "shape": shapes[0] if shapes else None,
        "dtype": str(first.dtype) if first is not None else None,
    }
