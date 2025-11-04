import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# GPU の自動初期化を行うモジュールである。
# import の副作用として CUDA デバイス選択とコンテキスト生成が実施されるため、
# 以降のカーネル起動・メモリ確保はこのデフォルトコンテキスト上で動作するである。
import pycuda.autoinit

# NVCC に渡されるホストコンパイラ（MSVC）向けフラグを環境変数で指定しているである。
# ここでは Windows 環境での日本語コードページ警告（4819）を抑制する意図である。
# Linux/GCC 環境では実質的な影響はないが、クロスプラットフォームの共通スクリプトとして保持していると解するである。
os.environ["CL"] = r'-Xcompiler "/wd 4819'


# CUDA C カーネルのヘッダ/実装を格納したディレクトリの絶対パスを解決するである。
# SourceModule に include_dirs を渡すことで、#include 参照を解決させる設計である。
cuda_file_path = os.path.abspath("./cuda")

#
#  CUDA カーネルの定義である。
#
# SourceModule は与えた CUDA C ソース（文字列）を NVCC で JIT コンパイルし、
# 生成された PTX/SASS をドライバ API を通じてロードするである。
# 本例では外部ファイル "kernel_functions_for_math_1d.cu" をインクルードしており、
# その中で __global__ void plus_one_kernel(...) が定義されている前提である。
# include_dirs に cuda_file_path を渡すことで、#include "..." の探索パスを追加している。
module = SourceModule("""
#include "kernel_functions_for_math_1d.cu"
""", include_dirs=[cuda_file_path])

# コンパイル済みモジュールからカーネル関数ハンドルを取得するである。
# 以後の起動時に block=(Bx,By,Bz), grid=(Gx,Gy,Gz) を明示指定して実行する。
plus_one_kernel = module.get_function("plus_one_kernel")

# 計算対象の NumPy 配列を生成するである。
# num_components は int32 として用意し、CUDA 側の int と型整合させる。
# x は 0..num_components-1 の連続整数配列（int32）である。
num_components = np.int32(10)
x = np.arange(num_components, dtype=np.int32)

# CPU→GPU 転送である。to_gpu は内部で cudaMalloc + cudaMemcpy(HostToDevice) 相当を実行し、
# デバイスポインタとメタ情報を保持する GPUArray を返す。
x_gpu = gpuarray.to_gpu(x)

# 出力用のデバイスメモリを確保するである。zeros は cudaMemset(0) 相当でゼロ初期化を行う。
y_gpu = gpuarray.zeros(num_components, dtype=np.int32)

# 実行構成（execution configuration）を決めるである。
# threads_per_block はブロック内スレッド数、blocks_per_grid は総要素数をカバーするよう切り上げ計算している。
# 端数スレッドはカーネル内の if(i<num_comp) により越境アクセスを防止する設計である。
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)

# CUDA カーネルの起動である。
# 引数は (num_comp, y, x) の順で渡され、GPUArray はデバイスポインタとして自動的に渡下される。
# カーネル起動は非同期であるが、この後の y_gpu.get() においてデバイス→ホスト転送時に同期が発生する。
plus_one_kernel(num_components, y_gpu, x_gpu, block=threads_per_block, grid=blocks_per_grid)

# GPU→CPU 転送である。get() は cudaMemcpy(DeviceToHost) を内部で呼び出し、完了までブロッキングする。
y = y_gpu.get()

# 検算用の出力である。期待される出力は x=[0..9], y=[1..10] である。
print("x :", x)
print("y :", y)