/*
 * 3次元スカラー場 arr(nx, ny, nz) の x 方向一次微分（勾配）を差分で評価し、
 * 結果を arr_grad に格納する CUDA グローバル関数である。
 *
 * 差分スキーム:
 *   - 内部点: 中心差分  (arr[x+1] - arr[x-1]) / (2*dx)
 *   - 左端点 x=0: 前進差分 (arr[x+1] - arr[x]) / dx
 *   - 右端点 x=nx-1: 後退差分 (arr[x] - arr[x-1]) / dx
 *
 * メモリレイアウト:
 *   3D 配列は行優先（row-major）で 1D 化されていると仮定し、
 *   線形添字 ijk = (z * ny + y) * nx + x = nx*ny*z + nx*y + x でアクセスする。
 *   よって x が最も速く変化する次元である（coalesced アクセスに有利）。
 *
 * 想定:
 *   - arr, arr_grad はデバイスメモリ上に確保された float 配列である。
 *   - dx は x 方向の格子間隔である（等間隔格子を仮定）。
 *   - グリッド/ブロックは 3D 構成で起動され、越境スレッドは if ガードで排除する。
 */
__global__ void calc_grad_x_3d(int nx, int ny, int nz, float dx, float *arr_grad, float *arr){
    // 各スレッドに割り当てられる 3D 座標 (x, y, z) を計算するである。
    // threadIdx.{x,y,z} : ブロック内スレッド座標
    // blockDim.{x,y,z}  : ブロック内スレッド数
    // blockIdx.{x,y,z}  : グリッド内ブロック座標
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    // 3D→1D の線形インデックスに変換するである（row-major）。
    // ijk    : 中心点 (x,y,z)
    // ijk_f  : x 方向の前方点 (x+1,y,z)
    // ijk_b  : x 方向の後方点 (x-1,y,z)
    int ijk   = nx * ny * z + nx * y + x;
    int ijk_f = nx * ny * z + nx * y + (x + 1);
    int ijk_b = nx * ny * z + nx * y + (x - 1);

    // 配列境界外のスレッドを排除するガードである。
    if (x < nx && y < ny && z < nz){
        // x 方向の勾配を差分で計算するである。
        // 端点では一方向差分、内部では中心差分を用いることで、
        // 配列外参照を避けつつ2次精度（中心差分）を内部で確保する。
        if (x == 0){
            // 左端点: 前進差分
            arr_grad[ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (x == (nx - 1)){
            // 右端点: 後退差分
            arr_grad[ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            // 内部点: 中心差分
            arr_grad[ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }
    }
}

/*
 * 最適化メモ:
 * - メモリアクセス: x が最速次元のため、threadIdx.x を x に対応させる現在の設計は
 *   読み書きが coalesced になりやすく帯域効率が良い。
 * - ブロック形状: blockDim.x を 32 の倍数（warp 整列）にするのが望ましい。
 *   例: (32,4,2) や (64,2,2) など。Occupancy とレジスタ使用量のバランスで調整する。
 * - 分岐: 端点条件の分岐は warp 内での分岐多様性を増やし得るが、端点は全体に対して少数であり、
 *   実害は小さいことが多い。必要に応じてゴーストセル導入で分岐除去も検討可能である。
 * - 精度: dx が空間的に一様でない場合はセル毎の dx(x) を配列で持たせる必要がある。
 * - 拡張: y,z 方向の勾配 calc_grad_y_3d / calc_grad_z_3d も同様のパターンで実装できる。
 */