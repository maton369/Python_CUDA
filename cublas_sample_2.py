import pycuda.autoinit
from pycuda import gpuarray
import numpy as np
# cublasのimport
from skcuda import cublas

# ------------------------------------------------------------
# 目的:
# cuBLAS の AXPY（SAXPY: Single-precision A*X + Y）を用いて
# ベクトル演算 y ← a * x + y を GPU 上で実行します。
# ・SAXPY は float32 専用の AXPY（“S” = single precision）です。
# ・結果は y に上書きされます（in-place 更新）。
# ------------------------------------------------------------

# スカラー a（float32）
a = np.float32(2)
# 入力ベクトル x, y（どちらも float32）
x = np.array([1, 2, 3], dtype=np.float32)
y = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Host → Device 転送（GPU メモリ上に連続領域として確保）
x_gpu = gpuarray.to_gpu(x)
y_gpu = gpuarray.to_gpu(y)

# 参照用の正解（CPU 側で計算）
print("answer :", a * x + y)

# cuBLAS ハンドル（コンテキスト）の作成
h = cublas.cublasCreate()

# ------------------------------------------------------------
# cuBLAS SAXPY の呼び出し
# cublasSaxpy(handle, n, alpha, x_devptr, incx, y_devptr, incy)
#   n      : 要素数
#   alpha  : スカラー a（float32）
#   x_dev  : デバイスポインタ（x の先頭）
#   incx   : x のストライド（連続なら 1）
#   y_dev  : デバイスポインタ（y の先頭）
#   incy   : y のストライド（連続なら 1）
# 実行結果は y 側に上書きされる（y ← alpha * x + y）。
# ------------------------------------------------------------
cublas.cublasSaxpy(h, x_gpu.size, a, x_gpu.gpudata, 1, y_gpu.gpudata, 1)

# cuBLAS ハンドルの破棄（リソース解放）
cublas.cublasDestroy(h)

# Device → Host 転送（計算結果の取得）
print("cublas axpy :", y_gpu.get())

# 補足:
# ・dtype は必ず float32（np.float32）に揃えてください。float64 の場合は DAXPY を使います。
# ・大規模データではホスト計算との誤差比較（相対/絶対誤差）も行うと安心です。
# ・PyCUDA のデフォルトストリームを使用しています。独自ストリームを使う場合はハンドルへ関連付けが必要です。
