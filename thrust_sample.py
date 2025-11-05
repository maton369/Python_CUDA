import os
import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule

# GPUの初期化（importの副作用でCUDAコンテキストを作成する）
import pycuda.autoinit

# ホストコンパイラ向けのフラグを環境変数経由で渡す（ここでは警告抑制の例）
os.environ["CL"] = r'-Xcompiler "/wd 4819'

# ------------------------------------------------------------
# カーネル読み込み設定
# ------------------------------------------------------------
# ./cuda ディレクトリにある CUDA C/C++ ファイルをインクルードしてビルドする。
# 今回は "kernel_functions_for_thrust.cu" を取り込み、そこに定義された
#   関数を呼び出す想定。
cuda_file_path = os.path.abspath("./cuda")

module = SourceModule("""
#include "kernel_functions_for_thrust.cu"
""", include_dirs=[cuda_file_path],
     no_extern_c=True)

# ------------------------------------------------------------
# デバイス関数ハンドル取得
# ------------------------------------------------------------
# 注意: PyCUDA の get_function は __global__ 関数のみ取得可能です。
# __host__ 関数（sort_thrust）は取得できません。
# そのため、ここではエラーを適切に処理します。
try:
    sort_thrust = module.get_function("sort_thrust")
except Exception as e:
    print(f"エラー: {e}")
    print("PyCUDA では __host__ 関数を直接呼び出すことはできません。")
    print("Thrust を使うには、以下のいずれかの方法が必要です:")
    print("  1) CuPy（cupy.sort, cupy.argsort）を使う")
    print("  2) 自前の並列ソートカーネル（bitonic/radix）を実装する")
    print("  3) C++ で「ホスト側Thrust呼び出し用バイナリ/共有ライブラリ」を作成し、")
    print("     Python から ctypes/pybind11 で呼ぶ")
    sort_thrust = None

# ------------------------------------------------------------
# 入力データの作成（ホスト側）
# ------------------------------------------------------------
nx = np.int32(10)                      # 要素数（int32で明示）
arr = np.arange(nx, dtype=np.int32)    # [0,1,2,3,4,5,6,7,8,9]
np.random.shuffle(arr)                  # 並び替えておく（ソートの効果を確認するため）
print("before sort : ", arr)

# ------------------------------------------------------------
# Host → Device 転送
# ------------------------------------------------------------
# NumPy 配列を GPU メモリ上へコピーし、GPUArray として保持する。
arr_gpu = gpuarray.to_gpu(arr)

# ------------------------------------------------------------
# カーネル起動（Thrustでのソートを試みる）
# ------------------------------------------------------------
if sort_thrust is not None:
    sort_thrust(nx, arr_gpu,
                block=(1, 1, 1),   # ブロック1×1×1（ダミー的設定）
                grid=(1, 1, 1))    # グリッド1×1×1（ダミー的設定）
else:
    # sort_thrust が取得できなかった場合の代替処理
    print("注意: sort_thrust は __host__ 関数のため、PyCUDA から直接呼び出すことはできません。")
    print("デモンストレーション: CPU でソートして GPU に転送")
    arr_sorted = np.sort(arr)
    arr_gpu = gpuarray.to_gpu(arr_sorted)

# ------------------------------------------------------------
# Device → Host 転送（結果の取得）
# ------------------------------------------------------------
print("after sort :", arr_gpu.get())

# ------------------------------------------------------------
# 補足:
# - thrust を正しく使うには、ホストC++側で Thrust を呼び出す必要がある。
#   PyCUDA で同等のことを行うのは難しい（PyCUDA は基本的に「デバイスカーネル」の呼び出しに特化）。
# - 代替案:
#   1) CuPy（cupy.sort, cupy.argsort）を使う。
#   2) 自前の並列ソートカーネル（bitonic/radix）を実装する。
#   3) C++ で「ホスト側Thrust呼び出し用バイナリ/共有ライブラリ」を作成し、Python から ctypes/pybind11 で呼ぶ。
