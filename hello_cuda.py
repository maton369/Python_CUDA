import numpy as np
import pycuda.gpuarray as gpuarray
from pycuda.elementwise import ElementwiseKernel

# GPU の自動初期化を行うモジュールである。
# import されるだけで CUDA デバイスの選択とコンテキスト作成が行われる（副作用）。
import pycuda.autoinit

#
#  CUDA カーネルの定義（要素単位カーネル）
#
# ElementwiseKernel は「配列の各要素 i に対して同一の処理を適用する」用途に特化したラッパである。
# 以下では int 型の配列 x の各要素に 1 を加算し、結果を y に書き込む。
# 第1引数: 引数の型シグネチャ（C 風の型指定）
# 第2引数: 実行される C コード断片（スレッドごとに y[i] = x[i] + 1 を評価）
# 第3引数: カーネル名（任意の識別用）
plus_one_kernel = ElementwiseKernel(
    "int *y, int *x",
    "y[i] = x[i] + 1",
    "plus_one"
)

# 計算対象の NumPy 配列を作成する。
# 0,1,2,...,num_components-1 の等差数列を 32bit 整数（int32）で用意する。
# ※ PyCUDA のカーネルで "int" を使う場合、ホスト側も np.int32 に合わせておくと型不一致を避けられる。
num_components = 10
x = np.arange(num_components, dtype=np.int32)

# CPU→GPU へのデータ転送を行う。
# to_gpu はホスト配列 x の内容を GPU メモリへコピーし、GPUArray オブジェクトを返す。
x_gpu = gpuarray.to_gpu(x)

# 出力用の GPU メモリを確保する。要素数は num_components、型は int32 である。
# 初期値は 0 で埋められる。
y_gpu = gpuarray.zeros(num_components, dtype=np.int32)

# CUDA カーネルを実行する。
# ElementwiseKernel は引数に GPUArray（あるいは互換の配列）を渡すだけで、
# 配列長に応じたグリッド/ブロック構成を自動的に決めて各要素に処理を適用する。
plus_one_kernel(y_gpu, x_gpu)

# GPU→CPU へのデータ転送を行う。
# GPU 側の y_gpu の内容をホスト側へ取り出し、NumPy 配列 y として受け取る。
y = y_gpu.get()

# 入出力の中身を表示する（検算用）。
print("x :", x)
print("y :", y)