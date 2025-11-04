import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# GPU の自動初期化である。import の副作用として CUDA デバイス選択と
# デフォルトコンテキスト生成が行われ、以降の API 呼び出しはこの文脈で実行されるである。
import pycuda.autoinit

# CUDA C カーネル群を配置したディレクトリの絶対パスを解決するである。
# SourceModule の include_dirs へ渡すことで #include 参照先を解決する設計である。
cuda_file_path = os.path.abspath("./cuda")

# NVCC に渡すホストコンパイラ用フラグを環境変数 CL で指定しているである。
# これは主に Windows/MSVC の警告抑制用途で、Linux/GCC 環境では実質影響しないである。
os.environ["CL"] = r'-Xcompiler "/wd 4819'

#
#  CUDA カーネルの定義である。
#
# 外部ファイル "kernel_functions_for_math_3d.cu" 内で 3D 勾配カーネル
#   __global__ void calc_grad_x_3d(int nx, int ny, int nz, float dx, float* grad, const float* arr)
# が定義されている前提である。ここではそれを #include し、NVCC で JIT コンパイル・ロードするである。
module = SourceModule("""
#include "kernel_functions_for_math_3d.cu"
""", include_dirs=[cuda_file_path])

# コンパイル済みモジュールからカーネル関数ハンドルを取得するである。
grad_x = module.get_function("calc_grad_x_3d")

# メッシュ生成のための 1D 座標配列を用意するである。
# 0≤x<4, 0≤y<6, 0≤z<8 を刻み幅 dx=0.01 で離散化し、float32 で保持する。
dx = np.float32(0.01)
x = np.arange(0, 4, dx, dtype=np.float32)
y = np.arange(0, 6, dx, dtype=np.float32)
z = np.arange(0, 8, dx, dtype=np.float32)

# 各次元の要素数を int32 として保持し、CUDA 側 int 引数と型整合させるである。
num_x = np.int32(len(x))
num_y = np.int32(len(y))
num_z = np.int32(len(z))
num_components = num_x * num_y * num_z  # 参考: 総要素数である（以降の計算には直接は使っていない）

# 3D グリッドを ijk（Z,Y,X の順）で生成するである。
# indexing="ij" により、Z.shape=(nz,ny,nx), Y.shape=(nz,ny,nx), X.shape=(nz,ny,nx) となる。
Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

# 計算対象スカラー場を定義するである。ここでは f(x,y,z)=x^2 を用いる。
# 期待される x 方向勾配は ∂f/∂x = 2x であるから、数値勾配の検算に適している。
arr = X ** 2  # 形状は (num_z, num_y, num_x)

# CPU→GPU 転送である。to_gpu は cudaMalloc + cudaMemcpy(HostToDevice) を内部で行う。
arr_gpu = gpuarray.to_gpu(arr)

# 出力勾配配列（x 方向）を GPU 上に確保するである。ゼロ初期化を行い、形状は入力と同一である。
arr_grad_x_gpu = gpuarray.zeros([num_z, num_y, num_x], dtype=np.float32)

# 実行構成（execution configuration）を決めるである。
# 3D ブロック (Bx,By,Bz) と 3D グリッド (Gx,Gy,Gz) を設定し、全要素をカバーする。
# 端数はカーネル側の境界チェックで保護される設計である。
threads_per_block = (6, 6, 6)
block_x = math.ceil(num_x / threads_per_block[0])
block_y = math.ceil(num_y / threads_per_block[1])
block_z = math.ceil(num_z / threads_per_block[2])
blocks_per_grid = (block_x, block_y, block_z)

# CUDA カーネルの起動である。
# 引数は (nx, ny, nz, dx, grad_out, arr_in) の順で渡し、block/grid を明示指定する。
# カーネル内部では (x,y,z) から row-major な線形添字 ijk を計算し、端点は一方向差分、
# 内部点は中心差分で ∂f/∂x を評価するである。
grad_x(num_x, num_y, num_z, dx, arr_grad_x_gpu, arr_gpu,
       block=threads_per_block, grid=blocks_per_grid)

# GPU→CPU 転送である。get() は cudaMemcpy(DeviceToHost) を内部で呼び出し、完了までブロッキングする。
arr_grad = arr_grad_x_gpu.get()

# 検算出力である。理論的には arr_grad ≈ 2*X となるはずである（差分誤差は O(dx^2)）。
print("result :", arr_grad)