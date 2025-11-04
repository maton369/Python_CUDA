# -*- coding: utf-8 -*-
"""
CUDA 12 対応：3D テクスチャ（cudaTextureObject_t）を Python（PyCUDA）から作成・利用するための補助関数群である。
レガシーなテクスチャ参照（texture<> / set_array / bind_to_texref）は CUDA 12 では非推奨・削除の流れであるため、
本ファイルでは **テクスチャオブジェクト** を生成する関数を提供する（推奨パス）である。

補足：
- 既存コードとの互換のため、ユーザー提示の `bind_array_to_texture3d`（テクスチャ参照向け）も残すが、
  コメントの通りレガシー API 用であり、CUDA 12 環境では動作しない／不安定な場合がある。
- 実運用では本モジュールの `make_cuda_array_3d` + `create_texobj_from_array_3d` を用い、
  カーネル側は `__global__ void kernel(cudaTextureObject_t texObj, ...)` として
  `tex3D<T>(texObj, u, v, w)` で読み出す設計へ移行するのが望ましいである。
"""

import numpy as np
import pycuda.driver as drv

# コンテキスト初期化（import 副作用）である。JIT/メモリ操作の前に必要である。
import pycuda.autoinit  # noqa: F401


# ============================================================
# レガシー API（参考・非推奨）：テクスチャ参照に 3D Array をバインドする関数である。
# ============================================================
def bind_array_to_texture3d(np_array: np.ndarray, tex_ref: drv.TextureReference):
    """
    【レガシー API / 非推奨】texture<> ベースの 3D テクスチャ参照にバインドするである。
    CUDA 12 ではレガシー参照は非推奨・削除のため、将来的な互換性は保証されない。
    代替としてテクスチャオブジェクト（create_texobj_from_array_3d）を用いるべきである。

    Parameters
    ----------
    np_array : np.ndarray
        形状 (D, H, W) の 3D 配列（C-order 前提）である。要素型は int32 など GPU 対応 dtype。
    tex_ref : drv.TextureReference
        PyCUDA のテクスチャ参照である（モジュールから get_texref("tex_3d") 等で取得）。

    Returns
    -------
    drv.Array
        バインドされた 3D cudaArray ハンドルである（寿命管理のため返す）。
    """
    # 形状の取得：C-order の (depth, height, width) である。
    d, h, w = np_array.shape

    # cudaArray の 3D 記述子を作成するである。
    descr = drv.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = drv.dtype_to_array_format(np_array.dtype)  # 要素フォーマットを dtype から導出
    descr.num_channels = 1
    descr.flags = 0  # 例：drv.array3d_flags.SURFACE_LDST 等を付与可能

    # 3D cudaArray を確保するである。
    device_array = drv.Array(descr)

    # 3D コピー（Host→Device）を行うである。ストライド情報をコピー器に渡す。
    copy = drv.Memcpy3D()
    copy.set_src_host(np_array)            # src = host pointer（連続領域想定、C-order）
    copy.set_dst_array(device_array)       # dst = cudaArray(3D)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]  # 1 行のバイト数（= W * itemsize）でよい
    copy.src_height = copy.height = h      # 行数
    copy.depth = d                         # スライス数
    copy()                                 # 転送実行

    # レガシー参照に 3D cudaArray をバインドするである（CUDA 12 では非推奨）。
    tex_ref.set_array(device_array)
    return device_array


# ============================================================
# 推奨 API：テクスチャオブジェクト（cudaTextureObject_t）を作成する関数である。
# ============================================================
def make_cuda_array_3d(np_array: np.ndarray) -> drv.Array:
    """
    3D cudaArray を確保し、NumPy 配列から 3D 転送して初期化するである（テクスチャオブジェクト用の下準備）。

    Parameters
    ----------
    np_array : np.ndarray
        形状 (D, H, W) の 3D 配列（C-order 前提）。

    Returns
    -------
    drv.Array
        初期化済みの 3D cudaArray である。
    """
    if not np_array.flags['C_CONTIGUOUS']:
        np_array = np.ascontiguousarray(np_array)

    d, h, w = np_array.shape

    descr = drv.ArrayDescriptor3D()
    descr.width = w
    descr.height = h
    descr.depth = d
    descr.format = drv.dtype_to_array_format(np_array.dtype)
    descr.num_channels = 1
    descr.flags = 0

    cu_array = drv.Array(descr)

    copy = drv.Memcpy3D()
    copy.set_src_host(np_array)
    copy.set_dst_array(cu_array)
    copy.width_in_bytes = copy.src_pitch = np_array.strides[1]
    copy.src_height = copy.height = h
    copy.depth = d
    copy()

    return cu_array


def create_texobj_from_array_3d(
    cu_array: drv.Array,
    *,
    read_mode: str = "element",          # "element" or "normalized_float"
    normalized_coords: bool = False,     # 非正規化座標（False）を推奨（(x+0.5,y+0.5,z+0.5)）
    filter_mode: str = "point",          # "point" or "linear"
    address_mode: tuple = ("clamp", "clamp", "clamp")  # 各軸のアドレッシング
) -> object:
    """
    3D cudaArray から cudaTextureObject_t を生成するである（CUDA 12 推奨パス）。

    Parameters
    ----------
    cu_array : drv.Array
        3D cudaArray ハンドルである。
    read_mode : {"element","normalized_float"}
        要素そのまま（int/float をそのまま）か、正規化 float で読むかである。
    normalized_coords : bool
        True で正規化座標（0..1）、False で非正規化座標（画素/ボクセル座標系）である。
    filter_mode : {"point","linear"}
        最近傍 or 線形補間である（int 読みのとき通常は point を推奨）。
    address_mode : (str,str,str)
        各軸の境界モード（"clamp","border","wrap","mirror" 等、PyCUDA が提供するものに依存）である。

    Returns
    -------
    drv.TextureObject
        作成されたテクスチャオブジェクトである（カーネルへ整数値として渡せる）。
    """

    # ---- enum の取得（PyCUDA のバージョン差異を吸収するために辞書化）である。
    # address mode
    am = getattr(drv, "address_mode", None)
    address_map = {
        "clamp": am.CLAMP if am else 0,
        "wrap": getattr(am, "WRAP", 1) if am else 1,
        "border": getattr(am, "BORDER", 2) if am else 2,
        "mirror": getattr(am, "MIRROR", 3) if am else 3,
    }
    # filter mode
    fm = getattr(drv, "filter_mode", None)
    filter_map = {
        "point": fm.POINT if fm else 0,
        "linear": getattr(fm, "LINEAR", 1) if fm else 1,
    }
    # read mode
    rm = getattr(drv, "read_mode", None)
    read_map = {
        "element": rm.ELEMENT_TYPE if rm else 0,
        "normalized_float": getattr(rm, "NORMALIZED_FLOAT", 1) if rm else 1,
    }
    # resource type
    rt = getattr(drv, "resource_type", None)
    resource_type_array = rt.ARRAY if rt else 0

    # ---- ResourceDesc（cudaResourceDesc 相当）：cudaArray を指定するである。
    res_desc = drv.ResourceDescriptor(resource_type_array, cuArr=cu_array)

    # ---- TextureDesc（cudaTextureDesc 相当）で各種モードを設定するである。
    tex_desc = drv.TextureDescriptor(
        address_modes=tuple(address_map.get(m, address_map["clamp"]) for m in address_mode),
        filter_mode=filter_map.get(filter_mode, filter_map["point"]),
        flags=0,
        read_mode=read_map.get(read_mode, read_map["element"]),
        sRGB=False,  # sRGB 無効
        normalized_coords=int(bool(normalized_coords)),
        max_anisotropy=1,
        mipmap_filter_mode=filter_map.get("point", 0),
        max_mipmap_level_clamp=0.0,
        min_mipmap_level_clamp=0.0,
        border_color=(0.0, 0.0, 0.0, 0.0),
    )

    # ---- TextureObject（cudaTextureObject_t 相当）を作成・返すである。
    TextureObjectCls = getattr(drv, "TextureObject", None)
    if TextureObjectCls is None:
        raise RuntimeError(
            "PyCUDA が TextureObject を提供していません。\n"
            "対処: 1) PyCUDA の更新、2) 代替としてレガシー参照APIを使用（CUDA 12 では非推奨）、\n"
            "3) 自前でドライバAPI経由のテクスチャオブジェクト作成ラッパを実装。"
        )
    tex_obj = TextureObjectCls(res_desc, tex_desc, None)
    return tex_obj


def destroy_texobj(tex_obj: object):
    """
    テクスチャオブジェクトを破棄するである（cudaDestroyTextureObject 相当）。
    """
    # PyCUDA の TextureObject は __del__ で解放されるが、明示破棄を推奨するである。
    tex_obj.destroy()


# ============================================================
# 使い方（概要）である：
# ============================================================
# -- カーネル（CUDA C 側）の例 --
# __global__ void read_texture_3d(cudaTextureObject_t texObj, int nx, int ny, int nz) {
#     int x = blockIdx.x * blockDim.x + threadIdx.x;
#     int y = blockIdx.y * blockDim.y + threadIdx.y;
#     int z = blockIdx.z * blockDim.z + threadIdx.z;
#     if (x < nx && y < ny && z < nz) {
#         // 非正規化座標 + 最近傍 の場合は (x+0.5, y+0.5, z+0.5) がボクセル中心である。
#         float u = x + 0.5f, v = y + 0.5f, w = z + 0.5f;
#         int val = tex3D<int>(texObj, u, v, w);
#         printf("(%d,%d,%d) -> %d\n", x, y, z, val);
#     }
# }
#
# -- Python（PyCUDA）側の最小シーケンス --
# arr3d = np.arange(D*H*W, dtype=np.int32).reshape(D,H,W)
# cu_array = make_cuda_array_3d(arr3d)
# tex_obj  = create_texobj_from_array_3d(cu_array, read_mode="element",
#                                        normalized_coords=False, filter_mode="point",
#                                        address_mode=("clamp","clamp","clamp"))
# func(tex_obj, np.int32(W), np.int32(H), np.int32(D), block=..., grid=...)
# destroy_texobj(tex_obj)
# cu_array.free()  # Array の寿命管理も忘れないである。
#
# 注意：
# - 本ファイルのレガシー関数 `bind_array_to_texture3d` は移行期間の便宜上のみ残している。
#   CUDA 12 完全移行後はテクスチャオブジェクト関数群の使用を推奨するである。
#
# 以上である。
if __name__ == "__main__":
    pass