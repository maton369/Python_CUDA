import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# GPU の自動初期化である。import の副作用として CUDA デバイス選択と
# デフォルトコンテキスト作成が行われ、以降の API 呼び出しはこの文脈で動作するである。
import pycuda.autoinit

# CUDA C カーネルの外部ファイルを格納したディレクトリの絶対パスを解決するである。
# SourceModule の include_dirs に渡すことで #include 参照先を解決させる設計である。
cuda_file_path = os.path.abspath("./cuda")

# NVCC に渡すホストコンパイラ向けオプションの環境変数である。
# ここでは Windows/MSVC の警告静穏化の例であり、Linux/GCC 環境では実質影響しないである。
os.environ["CL"] = r'-Xcompiler "/wd 4819'

#
#  CUDA カーネルの定義である。
#
# 外部ファイル "kernel_functions_for_math_2d.cu" 内に 2D 要素加算カーネル
#   __global__ void add_two_array_kernel(int nx, int ny, float* out, const float* a, const float* b)
# が実装されている前提である。ここではそのヘッダ/実装をインクルードし NVCC で JIT コンパイルするである。
module = SourceModule("""
#include "kernel_functions_for_math_2d.cu"
""", include_dirs=[cuda_file_path])

# コンパイル済みモジュールからカーネル関数ハンドルを取得するである。
# 起動時に block=(Bx,By,Bz), grid=(Gx,Gy,Gz) を明示指定して実行する。
add_two_array = module.get_function("add_two_array_kernel")

# 計算対象の NumPy 配列を作成するである。
# 注意: カーネルは (nx=列数, ny=行数) を仮定し、線形化を ij = nx*y + x としている。
# ここで num_x=5, num_y=2 としているため、理想的には配列形状は (ny, nx) = (2, 5) が一貫である。
# 本コードでは reshape(5, 2)（行=5, 列=2）となっており命名と形状が逆転しているが、
# 線形領域としては 10 要素で一致するため計算自体は成立する点に留意するである。
num_x, num_y = np.int32(5), np.int32(2)
num_components = num_x * num_y
x = np.arange(num_components, dtype=np.float32).reshape(5, 2)  # 教材的には reshape(num_y, num_x) が用語整合的である。
y = 10 * np.random.rand(5, 2)
y = np.float32(y)
res = np.zeros([5, 2], dtype=np.float32)

# CPU→GPU 転送である。to_gpu は cudaMalloc + cudaMemcpy(HostToDevice) を内部で行い、GPUArray を返す。
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)
res_gpu = gpuarray.to_gpu(res)  # 初期値ゼロで良ければ gpuarray.zeros(...) でもよいである。

# 実行構成（execution configuration）を決めるである。
# 2D ブロック (Bx,By,1) と 2D グリッド (Gx,Gy,1) を設定し、全要素 (nx,ny) をカバーする。
# Bx と By は warp 幅（32）の倍数で x 方向を優先的に広げるとメモリアクセスが coalesced になりやすいである。
threads_per_block = (16, 16, 1)
block_x = math.ceil(num_x / threads_per_block[0])  # Gx = ceil(nx / Bx)
block_y = math.ceil(num_y / threads_per_block[1])  # Gy = ceil(ny / By)
blocks_per_grid = (block_x, block_y, 1)

# CUDA カーネルの起動である。
# 引数は (nx, ny, out, a, b) の順で、GPUArray はデバイスポインタとして自動的に渡下される。
# カーネル内部では (x,y) = (threadIdx.x + blockDim.x*blockIdx.x, threadIdx.y + blockDim.y*blockIdx.y)
# を用い、境界チェック (x<nx && y<ny) により越境を防いでいるはずである。
add_two_array(num_x, num_y, res_gpu, x_gpu, y_gpu,
              block=threads_per_block, grid=blocks_per_grid)

# GPU→CPU 転送である。get() は cudaMemcpy(DeviceToHost) を内部で行い、完了までブロッキングする。
res = res_gpu.get()

# 検算出力である。期待値は res ≈ x + y（形状は作成時に与えた 5×2 のまま）である。
print("result :", res)

# --- 参考メモ（改善案, 実行時変更不要） ------------------------------------
# ・命名と形状の整合: 教材としては x.shape を (num_y, num_x) にすると説明と一致し誤解が減るである。
# ・ブロック形状: (32,8,1) や (16,16,1) など複数候補で性能実測し、coalesced の恩恵を確認すると良いである。
# ・大量データ: メモリ帯域律速の性質を見るには (4096,4096) など大きめの行列で評価するのが望ましいである。
# ・ページロック: 転送性能を上げるなら drv.pagelocked_empty を使った pinned メモリを検討するである。