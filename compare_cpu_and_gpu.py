import math
import os
import time
import numpy as np
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
from pycuda.compiler import SourceModule

# GPU の自動初期化である。import の副作用として CUDA デバイス選択と
# デフォルトコンテキスト作成が行われるため、以降の API 呼び出しは
# このコンテキスト上で実行される（明示的な init は不要）である。
import pycuda.autoinit

# NVCC に渡されるホストコンパイラ用オプションの環境変数である。
# ここでは Windows の MSVC 警告 4819 を抑制する例で、Linux/GCC では影響しない。
# 共通スクリプトを跨いだビルド静穏化の意図と解するである。
os.environ["CL"] = r'-Xcompiler "/wd 4819'


# CUDA C カーネル群の配置ディレクトリを絶対パスに解決するである。
# SourceModule に include_dirs を渡すことで、#include "..." の探索を可能にする。
cuda_file_path = os.path.abspath("./cuda")

#
#  CUDA カーネルの定義（外部 .cu をインクルードして JIT コンパイル）である。
#
# SourceModule は与えた CUDA C コードを NVCC でコンパイルし、PTX/SASS をロードする。
# この例では "kernel_functions_for_math_1d.cu" 内に __global__ plus_one_kernel が定義済みである前提である。
module = SourceModule("""
#include "kernel_functions_for_math_1d.cu"
""", include_dirs=[cuda_file_path])

# コンパイル済みモジュールからカーネル関数ハンドル（pycuda.driver.Function）を取得するである。
plus_one_kernel = module.get_function("plus_one_kernel")

# 計算対象の NumPy 配列を作成するである。
# num_components は int32 のスカラー（10^5）で、CUDA 側の int と型整合させている。
# arange(num_components) は 0..num_components-1 の等差数列を生成する。
num_components = np.int32(1e5)
x = np.arange(num_components, dtype=np.int32)

#
#  CPU での実行時間計測（ベースライン）である。
#
# time.time() は秒単位の壁時計時刻で、差分をミリ秒に換算して報告している。
time_start_cpu = time.time()
x = x + 1  # ベクトル化された要素加算（NumPy の ufunc、CPU 上で SIMD 最適化され得る）
time_end_cpu = time.time()

#
#  GPU での実行である。
#

# CPU→GPU 転送。to_gpu は cudaMalloc + cudaMemcpy(HostToDevice) を内部で行う。
x_gpu = gpuarray.to_gpu(x)

# 出力用デバイスバッファを確保し、ゼロ初期化（cudaMemset(0) 相当）する。
y_gpu = gpuarray.zeros(num_components, dtype=np.int32)

# 実行構成を決める。threads_per_block はブロック内スレッド数（ここでは 256）、
# blocks_per_grid は全要素をカバーできるよう切り上げる。
# 端数スレッドはカーネル側の if(i<num_comp) で越境防止される設計である。
threads_per_block = (256, 1, 1)
blocks_per_grid = (math.ceil(num_components / threads_per_block[0]), 1, 1)

# 計測用 CUDA イベントを用意するである。
# Event は GPU タイムライン上のタイムスタンプを取り、time_till でミリ秒差を得る。
time_start_gpu = drv.Event()
time_end_gpu = drv.Event()

time_start_gpu2 = drv.Event()
time_end_gpu2 = drv.Event()

# カーネル実行の前後をイベントで挟んで計測する。
time_start_gpu.record()  # ここでデバイス側の時刻を記録
# CUDA カーネル起動（非同期）。block, grid で実行構成を明示的に指定する。
plus_one_kernel(num_components, y_gpu, x_gpu, block=threads_per_block, grid=blocks_per_grid)
time_end_gpu.record()    # カーネル起動直後に印を打つ
time_end_gpu.synchronize()  # ホストに戻る前に end イベント到達を待って計測を確定させる

# 転送（デバイス→ホスト）の計測を別イベント区間で行う。
time_start_gpu2.record()
# GPU→CPU 転送。get() は cudaMemcpy(DeviceToHost) を内部で行い、完了までブロッキングする。
y = y_gpu.get()
time_end_gpu2.record()
time_end_gpu2.synchronize()

#
# 実行時間の比較である。
#
# CPU 側：time.time の差分（秒）→ ミリ秒換算
print("CPU calculation {0} [msec]".format(1000 * (time_end_cpu - time_start_cpu)))

# 注意：下記 gpu_exec の算出は「kernel 区間（start→end）」と
# 「start→転送完了(end_gpu2)」を足しており、カーネル時間が重複計上される。
# 純粋な「総 GPU 側コスト（kernel + memcpy）」を評価したい場合は
#   total = time_end_gpu.time_till(time_end_gpu2)  # end_gpu→end_gpu2 の差ではなく、
# ではなく、別途「kernel_ms = start→end」「copy_ms = start2→end2」を分けて足し合わせるのが適切である。
gpu_exec = time_start_gpu.time_till(time_end_gpu) + time_start_gpu.time_till(time_end_gpu2)
print("GPU calculation {0} [msec]".format(gpu_exec))

# カーネル実行時間（start→end）である。
print("kernel exec {0} [msec]".format(time_start_gpu.time_till(time_end_gpu)))

# 転送時間の表示ラベルであるが、現在の算出は（start→end_gpu2）で、
# カーネル時間も含んでしまう点に注意（純粋な memcpy 計時ではない）。
# 厳密には time_start_gpu2.time_till(time_end_gpu2) を用いるべきである。
print("memory copy {0} [msec]".format(time_start_gpu.time_till(time_end_gpu2)))


# 検算出力。x は CPU 側で +1 済み、y は GPU 側でさらに +1 された値である。
print("x :", x)
print("y :", y)