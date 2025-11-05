import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import DynamicSourceModule

# ------------------------------------------------------------
# 目的：
# - 「動的並列（Dynamic Parallelism）」を使う CUDA カーネルを
#   PyCUDA から呼び出す最小例。
# - 親カーネル（add_two_vector_dynamic）の中で、子カーネル（add_two_vector）を
#   <<<grid_, block_>>> 構成で起動し、要素ごとの和を計算する。
#
# 前提：
# - GPU の Compute Capability が 3.5 以上であること（動的並列が必要）。
# - ビルドに RDC（-rdc=true）が必要。PyCUDA では DynamicSourceModule を用いる。
#   なお、環境によっては NVCC オプションの追加設定が必要になる場合がある。
# ------------------------------------------------------------

# GPU の初期化（import の副作用で CUDA コンテキストが生成される）
import pycuda.autoinit

# ホストコンパイラ向けのフラグ（ここでは警告抑制の例）を環境変数経由で設定
os.environ["CL"] = r'-Xcompiler "/wd 4819'

# ------------------------------------------------------------
# カーネル読み込み
# ------------------------------------------------------------
# ./cuda ディレクトリに置いた .cu を取り込む。
# kernel_functions_for_dynamic_parallel.cu 側には以下が想定されている：
#   - __global__ void add_two_vector(int nx, float *arr1, float *arr2, float *res)
#   - __global__ void add_two_vector_dynamic(int *grid, int *block, int nx,
#                                            float *arr1, float *arr2, float *res)
#     親カーネル add_two_vector_dynamic の中で、子カーネル add_two_vector を起動する。
cuda_file_path = os.path.abspath("./cuda")

module = DynamicSourceModule("""
#include "kernel_functions_for_dynamic_parallel.cu"
""", include_dirs=[cuda_file_path])

# 親カーネル（デバイス側で子カーネルを起動する）を取得
add_two_vector_dynamic = module.get_function("add_two_vector_dynamic")

# ------------------------------------------------------------
# 入力データ準備（ホスト側）
# ------------------------------------------------------------
num_comp = np.int32(10)                         # 要素数（カーネル引数に合わせて int32）
arr1 = np.arange(num_comp, dtype=np.float32)    # 0..N-1 の等差数列
arr2 = np.arange(num_comp, dtype=np.float32)    # 0..N-1 の等差数列（シャッフルして順序を崩す）
np.random.shuffle(arr2)

# 出力用バッファ（GPU 側に確保：0 初期化）
res_gpu = gpuarray.zeros(num_comp, dtype=np.float32)

# ------------------------------------------------------------
# 実行構成（子カーネル用の grid/block）
# ------------------------------------------------------------
# ここで作る threads_per_block / blocks_per_grid は、
# 「親カーネルの引数として渡し、親が子カーネルを起動する際に用いる」ためのもの。
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_comp / threads_per_block[0]), 1, 1)

# 親カーネルに渡すため、grid/block を int[3] 形の配列として用意する（x, y, z の順）
block = np.array(threads_per_block, dtype=np.int32)
grid = np.array(blocks_per_grid, dtype=np.int32)

# ------------------------------------------------------------
# Host → Device 転送
# ------------------------------------------------------------
arr1_gpu = gpuarray.to_gpu(arr1)
arr2_gpu = gpuarray.to_gpu(arr2)
block_gpu = gpuarray.to_gpu(block)   # 親カーネル内で dim3(block[0], block[1], block[2]) に変換される前提
grid_gpu  = gpuarray.to_gpu(grid)    # 同上、dim3(grid[0], grid[1], grid[2])

# ------------------------------------------------------------
# 親カーネルを起動（デバイス側で子カーネルを起動する）
# ------------------------------------------------------------
# 注意：
# - 親カーネル自体は 1 ブロック 1 スレッドで十分（ここでは block=(1,1,1), grid=(1,1,1)）。
# - 親カーネルは内部で <<<grid_, block_>>> を使って子カーネルを実行する。
# - 既定では「親の終了＝子の完了」。ホスト側で全体完了を待つには cudaDeviceSynchronize() に相当する同期が必要。
add_two_vector_dynamic(grid_gpu, block_gpu, num_comp, arr1_gpu, arr2_gpu, res_gpu,
                       block=(1, 1, 1), grid=(1, 1, 1))

# ------------------------------------------------------------
# 検算（CPU 側）と結果の取得
# ------------------------------------------------------------
print("answer :", arr1 + arr2)   # 参照値（CPU）
print("result : ", res_gpu.get())  # GPU（動的並列経由で計算した結果）

# ------------------------------------------------------------
# メモ：
# - 動的並列はオーバーヘッドがあるため、単純な 1 パス処理では親から直接カーネルを起動したほうが速いことが多い。
#   動的並列は「データ依存で起動数が変わる再帰的処理」などで威力を発揮する。
# - 親カーネル内での同期（__syncthreads）はブロック内同期のみであり、子カーネルの完了待ちには関与しない点に注意。
# - 環境によっては NVCC オプション（-rdc=true, -lcudadevrt 等）の調整が必要になることがある。
# ------------------------------------------------------------