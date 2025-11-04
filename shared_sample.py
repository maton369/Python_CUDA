import math
import os
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# GPU の自動初期化を行うである。
# import の副作用として CUDA デバイス選択とデフォルトコンテキストの生成が行われ、
# 以降の API 呼び出しはこのコンテキスト上で実行されるである。
import pycuda.autoinit

# 外部 CUDA C カーネル(.cu)を配置したディレクトリの絶対パスを得るである。
# SourceModule の include_dirs に渡すことで #include の解決が可能になるである。
cuda_file_path = os.path.abspath("./cuda")

# NVCC に渡るホストコンパイラ用フラグを環境変数で指定しているである。
# これは主に Windows/MSVC の警告抑制用途であり、Linux/GCC 環境では実質的影響は小さいである。
os.environ["CL"] = r'-Xcompiler "/wd 4819'

#
#  CUDA カーネルの定義である。
#
# "kernel_functions_for_shared.cu" 内には以下の2カーネルが定義されている前提である:
#   - calc_grad_shared_3d : 共有メモリを用いた 3D 勾配計算（高速化版）
#   - calc_grad_global_3d : グローバルメモリを直接読む 3D 勾配計算（基準・シンプル版）
# SourceModule は文字列ソースを NVCC で JIT コンパイルし、PTX/SASS をロードするである。
module = SourceModule("""
#include "kernel_functions_for_shared.cu"
""", include_dirs=[cuda_file_path])

# コンパイル済みモジュールからカーネル関数ハンドルを取得するである。
grad_x_s = module.get_function("calc_grad_shared_3d")
grad_x_g = module.get_function("calc_grad_global_3d")

# カーネル実行時間計測用の CUDA イベントを用意するである。
# record() でデバイスタイムラインに印を打ち、time_till() でミリ秒差を取得できるである。
kernel_s = drv.Event()
kernel_e = drv.Event()

# ------------------------------------------------------------
# メッシュ生成と入力スカラー場の構築である。
# ------------------------------------------------------------
dx = np.float32(0.01)  # 格子間隔（x,y,z とも同一間隔と仮定）である。
x = np.arange(0, 4, dx, dtype=np.float32)
y = np.arange(0, 6, dx, dtype=np.float32)
z = np.arange(0, 8, dx, dtype=np.float32)

# 各次元の要素数を CUDA 側の int と整合させるため np.int32 で保持するである。
num_x = np.int32(len(x))
num_y = np.int32(len(y))
num_z = np.int32(len(z))
num_components = num_x * num_y * num_z  # 参考: 総セル数である。

# Z, Y, X の順で 3D メッシュを生成するである（indexing="ij" により形状は (nz, ny, nx) となる）。
Z, Y, X = np.meshgrid(z, y, x, indexing="ij")

# 計算対象の 3D スカラー場を定義するである。ここでは f(x,y,z)=x^2 とする。
# 理論勾配は ∂f/∂x = 2x, ∂f/∂y = 0, ∂f/∂z = 0 であり、検算に適するである。
arr = X ** 2  # 形状は (num_z, num_y, num_x) である。

# ------------------------------------------------------------
# デバイスメモリの確保とホスト→デバイス転送である。
# ------------------------------------------------------------
arr_gpu = gpuarray.to_gpu(arr)  # 入力スカラー場を GPU へコピーするである。
num_direction = np.int32(3)     # 勾配ベクトルの 3 方向 (x,y,z) を表すである。

# 出力配列 arr_grad_gpu は [方向, z, y, x] の 4 次元で確保するである。
# カーネル側は arr_grad 内で方向ごとに nxyz オフセットを足して書き込む設計であり、
# この 4D 形状は線形メモリ上では同じ並びになるため整合するである。
arr_grad_gpu = gpuarray.zeros([num_direction, num_z, num_y, num_x], dtype=np.float32)

# ------------------------------------------------------------
# 実行構成（execution configuration）の決定である。
# ------------------------------------------------------------
# 3D ブロック (Bx,By,Bz) と 3D グリッド (Gx,Gy,Gz) を設定し、全要素をカバーするである。
# 端数セルはカーネル内の境界チェックで保護されるである。
threads_per_block = (6, 6, 6)
block_x = math.ceil(num_x / threads_per_block[0])
block_y = math.ceil(num_y / threads_per_block[1])
block_z = math.ceil(num_z / threads_per_block[2])
blocks_per_grid = (block_x, block_y, block_z)

# ------------------------------------------------------------
# グローバルメモリ版カーネルの実行と計測である。
# ------------------------------------------------------------
kernel_s.record()  # 開始を記録するである。
grad_x_g(num_x, num_y, num_z, dx, arr_grad_gpu, arr_gpu,
         block=threads_per_block, grid=blocks_per_grid)
kernel_e.record()  # 終了を記録するである。
kernel_e.synchronize()  # 計測確定のため同期するである。
print("Global memory: {} [ms]".format(kernel_s.time_till(kernel_e)))

# 結果取得（必要に応じて検証用）。形状は (3, nz, ny, nx) である。
arr_global = arr_grad_gpu.get()

# ------------------------------------------------------------
# 共有メモリ版カーネルの実行と計測である。
# ------------------------------------------------------------
# 同一のイベントオブジェクトを再利用しているが、record→record→synchronize の順で
# 毎回新しい区間を測っているため問題ないである（デフォルトストリームを想定）。
kernel_s.record()
grad_x_s(num_x, num_y, num_z, dx, arr_grad_gpu, arr_gpu,
         block=threads_per_block, grid=blocks_per_grid)
kernel_e.record()
kernel_e.synchronize()
print("Shared memory: {} [ms]".format(kernel_s.time_till(kernel_e)))

# 結果取得（共有メモリ版）。以降の比較や検算に用いるである。
arr_shared = arr_grad_gpu.get()

# ------------------------------------------------------------
# 最終的な結果ダンプである（ここでは直前に書かれた内容が入る）。
# 実際の検証では arr_global と arr_shared の差分統計（L2 誤差など）を比較すると良いである。
# また理論解 2*X と比較することで数値誤差を評価できるである。
# ------------------------------------------------------------
arr_grad = arr_grad_gpu.get()
print("result :", arr_grad)