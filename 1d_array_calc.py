import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# GPU の自動初期化を行う（import の副作用で CUDA デバイス選択とコンテキスト生成がなされる）である。
import pycuda.autoinit

# NVCC に渡すホストコンパイラオプションを環境変数経由で指定しているである。
# ここでは Visual C++ の警告 4819 を抑制するための例（Windows/日本語コードページ向け）であり、
# Linux/GCC 環境では実質的な影響はない。クロスプラットフォームで共通スクリプトを使う意図と解する。
os.environ["CL"] = r'-Xcompiler "/wd 4819'

#
#  CUDA カーネルの定義（明示的な grid/block とインデックス計算を行う低レベル形式）である。
#
# SourceModule は与えた CUDA C コードを NVCC で JIT コンパイルし、PTX/SASS をロードする。
# ここで定義する plus_one_kernel は、全スレッドで一次元のグローバルインデックス i を計算し、
# 範囲内（i < num_comp）であれば y[i] = x[i] + 1 を実行する。
# - threadIdx.x : ブロック内のスレッド番号
# - blockDim.x  : ブロック内のスレッド数
# - blockIdx.x  : グリッド内のブロック番号
# → i = blockDim.x * blockIdx.x + threadIdx.x は 1D 配列を並列走査する典型式である。
module = SourceModule("""
    __global__ void plus_one_kernel(int num_comp, int *y, int *x){
       int i = threadIdx.x + blockDim.x * blockIdx.x;
       if (i < num_comp){
           y[i] = x[i] + 1;
        }
    }
""")

# コンパイル済みモジュールからカーネル関数ハンドル（pycuda.driver.Function）を取得するである。
# 呼び出し時に block=(Bx,By,Bz), grid=(Gx,Gy,Gz) を指定して起動する。
plus_one_kernel = module.get_function("plus_one_kernel")

# 計算対象の NumPy 配列を生成するである。
# num_components は int32 として用意し、CUDA カーネルの int 引数と型整合させている。
# x は 0..num_components-1 の連続整数を int32 配列で確保する。
num_components = np.int32(10)
x = np.arange(num_components, dtype=np.int32)

# CPU→GPU 転送である。to_gpu は cudaMalloc + cudaMemcpy(HostToDevice) 相当を内部で行い、
# デバイスポインタを保持する GPUArray を返す。
x_gpu = gpuarray.to_gpu(x)

# 出力用の GPU メモリを確保するである。zeros は cudaMemset(0) 相当でゼロ初期化する。
y_gpu = gpuarray.zeros(num_components, dtype=np.int32)

# 実行構成（execution configuration）を決めるである。
# threads_per_block は 1 ブロック内のスレッド数（最大は GPU ごとの上限に依存、一般に 1024 など）。
# blocks_per_grid は 総要素数 / blockDim.x を切り上げて計算し、全要素をカバーする。
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)  # 端数分は if(i<num_comp) で保護

# CUDA カーネルを起動するである。
# 引数は (num_comp, y, x) の順で、GPUArray はデバイスポインタとして自動的に渡される。
# 起動時に block/grid を明示指定することで、ElementwiseKernel と異なり完全に手動制御となる。
# 起動は非同期であるが、その後の y_gpu.get() は転送時に同期する（結果一貫性の担保）。
plus_one_kernel(num_components, y_gpu, x_gpu, block=threads_per_block, grid=blocks_per_grid)

# GPU→CPU 転送である。get() は cudaMemcpy(DeviceToHost) を内部で行い、完了までブロッキングする。
y = y_gpu.get()

# 検算出力である。期待される結果は x が [0..9]、y が [1..10] である。
print("x :", x)
print("y :", y)