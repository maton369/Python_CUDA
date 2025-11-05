import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda.driver as drv
import cuda_utils as cu  # 付随ユーティリティ（本スニペットでは直接は未使用だが、cudaArray 生成等で使う前提のヘルパ）

# GPU の初期化（import の副作用でデフォルトコンテキストが生成される）
import pycuda.autoinit

# NVCC に渡るホストコンパイラ向けオプションを環境変数で付与（警告抑制など）
os.environ["CL"] = r'-Xcompiler "/wd 4819'

# ------------------------------------------------------------
# カーネル読み込み
# ------------------------------------------------------------
# 外部 CUDA C ファイル（.cu）をインクルードしてビルドする。
# ここでは "kernel_functions_for_atomic.cu" 内に定義された
#   __global__ void sum_atomic(int nx, int *sum, int *data)
# を利用する想定。
cuda_file_path = os.path.abspath("./cuda")
module = SourceModule("""
#include "kernel_functions_for_atomic.cu"
""", include_dirs=[cuda_file_path])

# コンパイル済みモジュールからデバイス関数（グローバル関数）を取得
sum_kernel = module.get_function("sum_atomic")

# ------------------------------------------------------------
# 入力データの用意（ホスト側）
# ------------------------------------------------------------
nx = np.int32(10)                         # 要素数（int32 に明示）
arr = np.arange(nx, dtype=np.int32)       # [0,1,2,3,4,5,6,7,8,9]

# デバイス側の合計用バッファを用意（長さ1の int32 配列）
# atomicAdd の加算先（sum）として使う。ゼロ初期化される。
sum_gpu = gpuarray.zeros(1, dtype=np.int32)

# 入力配列を Host→Device 転送
arr_gpu = gpuarray.to_gpu(arr)

# ------------------------------------------------------------
# 実行構成（execution configuration）の決定
# ------------------------------------------------------------
# 1D カーネルなので x 次元のみ使用。
# スレッド数は 256/threadBlock、必要ブロック数は切り上げで算出する。
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(nx / threads_per_block[0]), 1, 1)

# ------------------------------------------------------------
# カーネル実行
# ------------------------------------------------------------
# sum_atomic のシグネチャ:
#   sum_atomic(int nx, int *sum, int *data)
# 各スレッドが data[x] を読み、atomicAdd(sum, data[x]) で単一カウンタに集計する。
# 競合は atomicAdd により直列化され正しさが保たれる。
sum_kernel(nx, sum_gpu, arr_gpu, block=threads_per_block, grid=blocks_per_grid)

# ------------------------------------------------------------
# 結果の検算と出力
# ------------------------------------------------------------
# CPU 側の参照値（np.sum）と、GPU 側のアトミック集計結果を比較表示する。
print("answer :", np.sum(arr))
print("atomic sum :", sum_gpu.get())

# ------------------------------------------------------------
# メモ:
# - この方式は要素ごとに atomicAdd を実行するため、要素数が大きいと競合でスループットが低下しやすい。
#   典型的な高速化は「ブロック内で共有メモリに部分和 → 各ブロック代表だけが 1 回 atomicAdd」を行う段階的リダクション。
# - オーバーフローが懸念される場合は 64bit（long long）型への変更と atomicAdd(long long*, long long) を用いる。
# - PyCUDA では gpuarray.zeros(1, dtype=np.int32) を「スカラー入出力バッファ」として扱うパターンが手軽。
# - カーネル側の境界条件 (x < nx) により、余分なスレッドが存在しても範囲外アクセスを防止できる。