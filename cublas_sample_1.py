import numpy as np
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
# cublasを使う為にimport
from skcuda import cublas

# ------------------------------------------------------------
# 目的（である）:
# - skcuda.cublas の i?amax 系関数を用いて、配列の最大要素のインデックスを GPU 上で取得する。
#   ※ 重要注意:
#     ・cuBLAS の `cublasIsamax` は **float32（単精度実数）配列**に対する
#       「絶対値が最大の要素の **1始まり** インデックス」を返す関数である。
#       本コードは int32 配列を与えているため、**型不一致**で未定義動作・誤結果となる恐れがある。
#       実務では float32 へ明示変換（例: x.astype(np.float32)）するか、
#       整数対応の独自カーネルや Thrust/CUB 等を用いるべきである。
#     ・NumPy の np.argmax は **0始まり** インデックスを返すため、結果を比較する際は
#       `cublas` の戻り値から 1 を減算して照合すること（である）。
# ------------------------------------------------------------

# 最大値を求める用のnumpyアレー（乱数で生成）である。
np.random.seed(123)
x = np.random.randint(1, 100, 20, dtype=np.int32)

# Host→Device 転送である。GPU 側に int32 の連続配列として配置される。
x_gpu = gpuarray.to_gpu(x)

# 正解を出力（CPU側のゼロ始まりargmax）である。
print("target numpy array is : ", x)
print("answer : ", np.argmax(x))

# ------------------------------------------------------------
# cuBLAS で最大値のインデックスを求める（である）。
# `cublasCreate` でハンドル（コンテキスト）を作成し、`cublasDestroy` で破棄する。
# `cublasIsamax(handle, n, x_devptr, incx)` は
#   - n: 要素数
#   - x_devptr: デバイスポインタ（float* を想定）
#   - incx: ストライド。連続なら 1
# を与え、**1始まり**のインデックス（int）を返す。
# ここで int32 配列を渡している点は要注意（型不一致の可能性）である。
# ------------------------------------------------------------
h = cublas.cublasCreate()
max_id = cublas.cublasIsamax(h, x_gpu.size, x_gpu.gpudata, 1)
cublas.cublasDestroy(h)
print("max id based on cublas : ", max_id)

# ------------------------------------------------------------
# 検算の指針（である）:
# - NumPy の np.argmax(x)（0始まり） と比較するには、
#     (max_id - 1) == np.argmax(x)
#   を確認すること。
# - 正式には x を float32 へ変換し、
#     x_f = x.astype(np.float32); x_f_gpu = gpuarray.to_gpu(x_f)
#   とした上で cublasIsamax を適用するのが正しい（である）。
# ------------------------------------------------------------
