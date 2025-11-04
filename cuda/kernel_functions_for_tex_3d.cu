/*
 * CUDA 12 準拠（レガシー texture<> 非使用）の 3D テクスチャ読み出し最小例である。
 * 目的：
 *   - 3D 配列（nx×ny×nz, 要素型 int）を 3D cudaArray に転送し、
 *   - テクスチャオブジェクト（cudaTextureObject_t）を生成して、
 *   - カーネルで tex3D<int>(...) により要素を読み出す。
 *
 * 重要ポイント（である）:
 * - **レガシー `texture<T,3,...>` は使わず**、ホスト側で `cudaTextureObject_t` を作成し、
 *   それを **カーネル引数** として渡すのが CUDA 12 の流儀である。
 * - 3D テクスチャは通常 **cudaArray (3D)** を用いる（pitch3D リソースでもよいが、配列型が素直である）。
 * - 非正規化座標（normalizedCoords=0）かつ最近傍サンプリング（Point）の場合、
 *   画素・ボクセル中心をサンプルするために `(x+0.5f, y+0.5f, z+0.5f)` を渡すのが定石である。
 */

#include <cuda_runtime.h>
#include <cstdio>

// （PyCUDA からインクルードされる想定のため、ホスト側ユーティリティや STL には依存しない）

/*
 * 3D テクスチャから int を読み取るカーネルである。
 * 引数：
 *   texObj : ホスト側で生成した cudaTextureObject_t
 *   nx,ny,nz : 有効ボクセル数（範囲チェック用）
 *
 * 座標系：
 *   normalizedCoords=0（非正規化座標）のため、テクセル中心に合わせるべく 0.5 を加えるである。
 */
__global__ void read_texture_3d(cudaTextureObject_t texObj, int nx, int ny, int nz) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // 0..nx-1
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // 0..ny-1
    int z = blockIdx.z * blockDim.z + threadIdx.z;  // 0..nz-1
    if (x < nx && y < ny && z < nz) {
        float u = static_cast<float>(x) + 0.5f;
        float v = static_cast<float>(y) + 0.5f;
        float w = static_cast<float>(z) + 0.5f;

        // 要素型 int をそのまま取得（cudaReadModeElementType）である。
        int value = tex3D<int>(texObj, u, v, w);
        printf("x: %d, y: %d, z: %d, my value is %d\n", x, y, z, value);
    }
}

/*
 * ビルド例（である）:
 *   nvcc -std=c++14 -O2 -arch=sm_80 tex3d_example.cu -o tex3d_example
 *   ./tex3d_example
 *
 * 検算の考え方（である）:
 *   - 出力は "x: i, y: j, z: k, my value is v" の形式であり、
 *     v = (k*10000 + j*100 + i) と一致するはずである。
 *   - normalizedCoords=0・Point のため、(x+0.5, y+0.5, z+0.5) で中心サンプルを行っている。
 *
 * 応用（である）:
 *   - 3D テクスチャは体積データ（CT/MRI, CFD）に有効である。
 *   - 線形補間が必要な場合は filterMode=Linear とし、フォーマット/テクスチャ設定の制約に留意すること。
 *   - PyCUDA から扱う場合はドライバ API を介して cudaArray/cudaTextureObject_t を生成・受け渡しする
 *     ラッパを用意するのが実務的である（レガシー参照 API は CUDA 12 では非推奨である）。
 */