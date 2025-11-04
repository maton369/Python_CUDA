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

# CUDA C カーネルファイル群を配置したディレクトリの絶対パスを解決するである。
cuda_file_path = os.path.abspath("./cuda")

#
#  CUDA カーネルの定義である。
#
# 外部ファイル "kernel_functions_for_tex_1d.cu" 内には、
#   - レガシー API の 1D テクスチャ参照 `texture<int,1,cudaReadModeElementType> tex_1d;`
#   - その読み出しカーネル `__global__ void read_texture_1d(int nx)`
# が定義されている前提である。ここではそれを #include し NVCC で JIT コンパイルするである。
module = SourceModule("""
#include "kernel_functions_for_tex_1d.cu"
""", include_dirs=[cuda_file_path])

# コンパイル済みモジュールからカーネル関数ハンドルを取得するである。
read_tex_1d = module.get_function("read_texture_1d")

# 計算対象の NumPy 配列を作成するである。0..num_components-1 の等差列を int32 で生成する。
num_components = np.int32(10)
x = np.arange(num_components, dtype=np.int32)

# テクスチャキャッシュの動作確認のために要素順をランダム化するである。
# これによりコアレス化とは異なるパターンでも tex1Dfetch によるヒットを観察できる可能性がある。
np.random.shuffle(x)

# 正解データ（ホスト側配列）を表示するである。カーネル側の printf 出力と突き合わせる検算用である。
print(x)

# ホスト→デバイス転送である。x_gpu は `GPUArray<int32>` を指し、
# 内部で cudaMalloc と cudaMemcpy(HostToDevice) が行われるである。
x_gpu = gpuarray.to_gpu(x)

# ------------------------------------------------------------
# テクスチャ参照の取得とバインドである。
# ------------------------------------------------------------
# レガシーなテクスチャ参照（module-scope の `texture<>` 変数）を Python 側から参照するである。
# 注意: CUDA 12.0以降ではレガシーテクスチャ参照APIが削除されているため、
# get_texrefは動作しない可能性があります。CUDA 11.x以前を使用するか、
# 新しいテクスチャオブジェクトAPIに移行する必要があります。
try:
    tex_1d = module.get_texref("tex_1d")
except drv.LogicError as e:
    if "named symbol not found" in str(e):
        raise RuntimeError(
            "CUDA 12.0以降ではレガシーテクスチャ参照APIが削除されています。\n"
            "解決策:\n"
            "1. CUDA 11.x以前を使用する（推奨）\n"
            "2. 新しいテクスチャオブジェクトAPI（cudaTextureObject_t）に移行する\n"
            "詳細: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#deprecated-features"
        ) from e
    raise

# デバイスメモリ（線形メモリ）をテクスチャ参照にバインドするである。
# bind_to_texref_ext は内部で cudaBindTexture と等価の処理を行い、
# カーネル内から tex1Dfetch(tex_1d, idx) により読み出せるようにする。
x_gpu.bind_to_texref_ext(tex_1d)

# ------------------------------------------------------------
# 実行構成（execution configuration）の決定である。
# ------------------------------------------------------------
# 1D カーネルであるため (Bx,1,1) のブロックと (Gx,1,1) のグリッドで起動する。
# 端数スレッドはカーネル側の境界チェック (x<nx) で保護されるである。
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)

# ------------------------------------------------------------
# CUDA カーネルの実行である。
# ------------------------------------------------------------
# 第1引数 nx はテクスチャの有効長（配列長）を表し、範囲チェックに用いられるである。
# texrefs=[tex_1d] を渡すことで、この起動におけるテクスチャ参照のバインディングを確定させる。
# カーネル内では tex1Dfetch(tex_1d, x) により int 値が読み出され、printf で表示されるである。
read_tex_1d(num_components, block=threads_per_block, grid=blocks_per_grid, texrefs=[tex_1d])