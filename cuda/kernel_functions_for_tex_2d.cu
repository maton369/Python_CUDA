/*
 * 2次元テクスチャメモリから値を読み取る CUDA カーネルの例である。
 *
 * 背景と要点：
 * - CUDA 12.0以降では、新しいテクスチャオブジェクトAPI（`cudaTextureObject_t`）を使用する。
 * - `tex2D<int>(texObj, u, v)` により、テクスチャオブジェクトから値を読み取る。
 * - 非正規化座標を使用する場合、画素中心 (x+0.5, y+0.5) を指定する。
 */

#include <cuda_runtime.h>
#include <cstdio>

// 2D テクスチャから int を読み取るカーネル（CUDA 12 準拠、テクスチャオブジェクト版）である。
__global__ void read_texture_2d(cudaTextureObject_t texObj, int nx, int ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 0 <= x < nx
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 0 <= y < ny
    if (x < nx && y < ny) {
        // 非正規化座標 + Point サンプリングでは画素中心 (x+0.5, y+0.5) を指定するである。
        float u = static_cast<float>(x) + 0.5f;
        float v = static_cast<float>(y) + 0.5f;
        int value = tex2D<int>(texObj, u, v);
        printf("x: %d, y: %d, my value is %d\n", x, y, value);
    }
}