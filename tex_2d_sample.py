import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv

# GPU の自動初期化である。import の副作用として CUDA デバイス選択と
# デフォルトコンテキスト生成が行われ、以降の API 呼び出しはこの文脈で実行されるである。
import pycuda.autoinit

# NVCC に渡るホストコンパイラ用フラグを環境変数で指定するである。
# ここでは主に Windows/MSVC の警告抑制の例であり、Linux/GCC 環境では実質影響は小さいである。
os.environ["CL"] = r'-Xcompiler "/wd 4819'

# 外部 CUDA C カーネル(.cu)を配置したディレクトリの絶対パスを得るである。
cuda_file_path = os.path.abspath("./cuda")

#
#  CUDA カーネルの定義である。
#
# "kernel_functions_for_tex_2d.cu" 内には、CUDA 12 準拠のテクスチャオブジェクトAPIを使用する
# カーネル read_texture_2d(cudaTextureObject_t texObj, int nx, int ny) が定義されている前提である。
# 本コードは **新しいテクスチャオブジェクトAPI** を使用する。
module = SourceModule("""
#include "kernel_functions_for_tex_2d.cu"
""", include_dirs=[cuda_file_path])

# コンパイルしたモジュールからカーネル関数（グローバル関数）を取得するである。
read_tex_2d = module.get_function("read_texture_2d")

# ------------------------------------------------------------
# 入力データ（ホスト側 NumPy 配列）の作成である。
# ------------------------------------------------------------
num_components = np.int32(20)                          # 総要素数である。
x = np.arange(num_components, dtype=np.int32)          # 0..19 の等差列（int32）を用意するである。

# テクスチャキャッシュの挙動確認のため、要素順をランダム化しておくである。
np.random.shuffle(x)

# 2D 配列サイズを指定し、(num_y, num_x) 形状に整形するである。
num_x, num_y = np.int32(5), np.int32(4)                # 幅=5, 高さ=4 とするである（5*4=20）。
x = x.reshape(num_y, num_x)                            # C 行優先（row-major）で 2D 化するである。

# 正解データ（ホスト側）を表示しておくである。カーネル側の printf 出力と突き合わせる検算用である。
print(x)

# ホスト→デバイスの線形コピーである。x_gpu は GPUArray<int32>(shape=(num_y,num_x)) を指すである。
x_gpu = gpuarray.to_gpu(x)

# ------------------------------------------------------------
# テクスチャオブジェクトの作成（CUDA 12 準拠の新しいAPI）。
# ------------------------------------------------------------
# 2Dピッチ付きメモリの確保（CUDA 12ではテクスチャオブジェクトAPIを使用）
# mem_alloc_pitchは (device_ptr, pitch_bytes) のタプルを返す
d_img, pitch_bytes = drv.mem_alloc_pitch(
    num_x * np.dtype(np.int32).itemsize,
    num_y,
    np.dtype(np.int32).itemsize
)

# ホストデータをデバイスにコピー（ピッチ付き）
# x_gpuからピッチ付きメモリにコピー
drv.memcpy_2d(
    d_img, pitch_bytes,  # デスティネーション
    x_gpu.gpudata, num_x * np.dtype(np.int32).itemsize,  # ソース
    num_x * np.dtype(np.int32).itemsize, num_y,  # 幅と高さ
    drv.memcpy_device_to_device
)

# チャネルフォーマット記述子を作成（int32用）
channel_desc = drv.ChannelFormatDescriptor(
    np.dtype(np.int32).itemsize * 8, 0, 0, 0, drv.ChannelFormatKind.SIGNED
)

# テクスチャリソース記述子を作成
res_desc = drv.ResourceDescriptor()
res_desc.set_pitch2d(d_img, channel_desc, num_x, num_y, pitch_bytes)

# テクスチャ記述子を作成
tex_desc = drv.TextureDescriptor()
tex_desc.set_address_mode(0, drv.address_mode.CLAMP)
tex_desc.set_address_mode(1, drv.address_mode.CLAMP)
tex_desc.set_filter_mode(drv.filter_mode.POINT)
tex_desc.set_read_mode(drv.ReadMode.ELEMENT_TYPE)
tex_desc.set_normalized_coords(False)

# テクスチャオブジェクトを作成
tex_obj = drv.TextureObject(res_desc, tex_desc)

# ------------------------------------------------------------
# 実行構成（execution configuration）の決定である。
# ------------------------------------------------------------
# 2D カーネルであるため (Bx,By,1) のブロックと (Gx,Gy,1) のグリッドを設定するである。
threads_per_block = (16, 16, 1)
block_x = math.ceil(num_x / threads_per_block[0])      # 幅方向のブロック数である。
block_y = math.ceil(num_y / threads_per_block[1])      # 高さ方向のブロック数である。
blocks_per_grid = (block_x, block_y, 1)

# ------------------------------------------------------------
# CUDA カーネルの実行である。
# ------------------------------------------------------------
# read_texture_2d(texObj, nx, ny) は、各スレッドが (x,y) 画素に対応し、
# tex2D<int>(texObj, u, v) で値を読む。テクスチャオブジェクトを引数として渡す。
read_tex_2d(tex_obj, num_x, num_y, block=threads_per_block, grid=blocks_per_grid)

# 後片付け
tex_obj = None
drv.mem_free(d_img)

# 参考（モダンAPIへの移行指針・である）:
# - 本コードは互換のためレガシー参照を使用しているが、CUDA 12 準拠へ最終移行する場合は
#   * cudaMallocPitch + cudaMemcpy2D でデバイス配列を確保・転送
#   * cudaResourceDesc (pitch2D or array) と cudaTextureDesc を設定
#   * cudaCreateTextureObject で cudaTextureObject_t を生成
#   * カーネル引数に texObj を渡し、tex2D<T>(texObj, u, v) を呼び出す
#   という形に置き換えるのが推奨である。