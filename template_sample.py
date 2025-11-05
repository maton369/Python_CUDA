import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# ------------------------------------------------------------
# 目的:
# - テンプレート化された CUDA デバイス関数（int 向けに明示インスタンス化）を呼び出し、
#   1 次元ベクトル x, y の要素ごとの和を GPU 上で計算して res に格納します。
# - PyCUDA の SourceModule で外部 .cu を取り込み、__global__ カーネルを起動します。
# ------------------------------------------------------------

# GPU の初期化（import の副作用でデフォルトコンテキストが生成される）
import pycuda.autoinit

# ホストコンパイラへ渡すフラグ（ここでは警告抑制例）を環境変数経由で付与
os.environ["CL"] = r'-Xcompiler "/wd 4819'

# ------------------------------------------------------------
# カーネル読み込み
# ------------------------------------------------------------
# ./cuda ディレクトリに配置した CUDA C/C++ ファイルをインクルードしてビルドします。
# この中には次のカーネルが定義されている想定です:
#   extern "C" __global__
#   void add_two_vector_kernel(int nx, int *a, int *b, int *res)
cuda_file_path = os.path.abspath("./cuda")

module = SourceModule("""
#include "kernel_functions_for_template.cu"
""", include_dirs=[cuda_file_path],
     # no_extern_c=True: PyCUDA が自動で extern "C" を付けるのを抑止します
     # （.cu 側で明示している場合や C++ テンプレート都合で必要な場合に指定）
     no_extern_c=True)

# コンパイル済みモジュールから __global__ 関数のハンドルを取得
add_two_vector = module.get_function("add_two_vector_kernel")

# ------------------------------------------------------------
# 入力データの用意（ホスト側）
# ------------------------------------------------------------
np.random.seed(123)                                  # 再現性のため乱数シード固定
num_components = np.int32(10)                        # 要素数（型はカーネル引数に合わせて int32）
x = np.arange(num_components, dtype=np.int32)        # 0..N-1 の等差数列
y = np.random.randint(0, 10, num_components, dtype=np.int32)  # 0..9 の乱数列

# ------------------------------------------------------------
# Host → Device 転送
# ------------------------------------------------------------
# NumPy 配列を GPU メモリへコピーし、GPUArray（PyCUDA ラッパ）として保持します。
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
res_gpu = gpuarray.zeros(num_components, dtype=np.int32)  # 出力用（0 初期化）

# ------------------------------------------------------------
# 実行構成（execution configuration）の決定
# ------------------------------------------------------------
# 1D カーネルの典型: 1 ブロックあたり 256 スレッド、必要ブロック数は切り上げで算出します。
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)

# ------------------------------------------------------------
# カーネル起動
# ------------------------------------------------------------
# シグネチャ:
#   add_two_vector_kernel(int nx, int *a, int *b, int *res)
# 各スレッドは自分のインデックス x を計算し、範囲内で a[x] + b[x] を res[x] に書き込みます。
add_two_vector(num_components, x_gpu, y_gpu, res_gpu,
               block=threads_per_block, grid=blocks_per_grid)

# ------------------------------------------------------------
# Device → Host 転送（結果の取得）
# ------------------------------------------------------------
res = res_gpu.get()

# ------------------------------------------------------------
# 検算（CPU vs GPU）
# ------------------------------------------------------------
print("answer :", x + y)          # CPU 側での正解（要素ごとの加算）
print("result :", res_gpu.get())  # GPU 側の結果（上で res に取り出しているが可視化のため再取得）

# ------------------------------------------------------------
# メモ:
# - ブロック・グリッドの構成はデータサイズに応じて調整できます（例: 128/256/512 スレッドなど）。
# - 型を変えたい場合は .cu 側のテンプレート呼び出し（add_two_vector<T>）と
#   カーネルのポインタ型（int* → float* など）を合わせて変更します。
# - 範囲外アクセスはカーネル内の if (x < nx) ガードで防止されます。
# ------------------------------------------------------------