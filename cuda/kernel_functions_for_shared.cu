/*
 * 3次元スカラー場 arr(nx, ny, nz) の勾配（x, y, z 方向）を計算する CUDA カーネル群である。
 * 
 * 目的：
 *   - calc_grad_shared_3d : 共有メモリ(shared memory)を用いた勾配計算（高速版）
 *   - calc_grad_global_3d : グローバルメモリ(global memory)を直接参照する勾配計算（単純版・基準比較用）
 *
 * 定義済みマクロ：
 *   NUM_THREADS : 各ブロックのスレッド数（x, y, z 各方向）
 *   NUM_HALO    : 周辺のハロー（境界セル）数。NUM_THREADS+NUM_HALO により shared memory の拡張領域を確保。
 *   X_DIRECTION, Y_DIRECTION, Z_DIRECTION : 各方向インデックス。arr_grad 配列中で方向ごとに nxyz 要素分のオフセットを表す。
 *
 * 背景：
 *   共有メモリを利用することで、隣接セルへのアクセスを高速化し、
 *   グローバルメモリアクセスを減らすことができる（帯域効率向上）。
 *   同時に、ハロー領域を明示的にロードし、境界の値を共有メモリに保持することで
 *   各スレッドが独立して差分を計算できる。
 */

#define NUM_THREADS 6
#define X_DIRECTION 0
#define Y_DIRECTION 1
#define Z_DIRECTION 2
#define NUM_HALO 2

__global__ void calc_grad_shared_3d(int nx, int ny, int nz, float dx, float *arr_grad, float *arr){
    // ------------------------------
    // 共有メモリ領域の定義
    // ------------------------------
    // arr_s[z][y][x] 形式でブロック内に配置。
    // ハローセルを考慮し NUM_THREADS+NUM_HALO 分を確保している。
    __shared__ float arr_s[NUM_THREADS+NUM_HALO][NUM_THREADS+NUM_HALO][NUM_THREADS+NUM_HALO];

    // グローバル座標（全体空間におけるセルインデックス）
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    // ローカル座標（共有メモリ空間での位置）
    // 周囲1セル分のハローを考慮するため +1 オフセットを持つ。
    const int tx = threadIdx.x + 1;
    const int ty = threadIdx.y + 1;
    const int tz = threadIdx.z + 1;

    // 総要素数（arr_grad内で方向分割オフセットを計算するために利用）
    const int nxyz = nx * ny * nz;

    // ------------------------------
    // グローバルメモリ→共有メモリ コピー
    // ------------------------------
    if (x < nx && y < ny && z < nz){
        int ijk = nx * ny * z + nx * y + x;
        int ijk_f, ijk_b;

        // 中心セルを共有メモリに格納
        arr_s[tz][ty][tx] = arr[ijk];

        // ----------------------------------------
        // Halo領域のロード（X方向）
        // ----------------------------------------
        if (!(x == 0) && (tx == 1)){
            // 左側（ブロック境界の1セル内側）の場合、1セル前方をロード
            ijk_b = nx * ny * z + nx * y + (x - 1);
            arr_s[tz][ty][tx-1] = arr[ijk_b];
        } else if (!(x == 0) && (tx == NUM_THREADS)){
            // 右側（ブロック境界の1セル内側）の場合、1セル後方をロード
            ijk_f = nx * ny * z + nx * y + (x + 1);
            arr_s[tz][ty][tx+1] = arr[ijk_f];
        }

        // ----------------------------------------
        // Halo領域のロード（Y方向）
        // ----------------------------------------
        if (!(y == 0) && (ty == 1)){
            ijk_b = nx * ny * z + nx * (y - 1) + x;
            arr_s[tz][ty-1][tx] = arr[ijk_b];
        } else if (!(y == 0) && (ty == NUM_THREADS)){
            ijk_f = nx * ny * z + nx * (y + 1) + x;
            arr_s[tz][ty+1][tx] = arr[ijk_f];
        }

        // ----------------------------------------
        // Halo領域のロード（Z方向）
        // ----------------------------------------
        if (!(z == 0) && (tz == 1)){
            ijk_b = nx * ny * (z - 1) + nx * y + x;
            arr_s[tz-1][ty][tx] = arr[ijk_b];
        } else if (!(z == 0) && (tz == NUM_THREADS)){
            ijk_f = nx * ny * (z + 1) + nx * y + x;
            arr_s[tz+1][ty][tx] = arr[ijk_f];
        }

        // 全スレッドでデータコピーを完了するまで同期
        __syncthreads();

        // ------------------------------
        // X方向勾配の計算
        // ------------------------------
        if (x == 0){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr_s[tz][ty][tx+1] - arr_s[tz][ty][tx]) / dx;
        } else if (x == (nx - 1)){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr_s[tz][ty][tx] - arr_s[tz][ty][tx-1]) / dx;
        } else {
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr_s[tz][ty][tx+1] - arr_s[tz][ty][tx-1]) / (2.0 * dx);
        }

        // ------------------------------
        // Y方向勾配の計算
        // ------------------------------
        if (y == 0){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr_s[tz][ty+1][tx] - arr_s[tz][ty][tx]) / dx;
        } else if (y == (ny - 1)){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr_s[tz][ty][tx] - arr_s[tz][ty-1][tx]) / dx;
        } else {
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr_s[tz][ty+1][tx] - arr_s[tz][ty-1][tx]) / (2.0 * dx);
        }

        // ------------------------------
        // Z方向勾配の計算
        // ------------------------------
        if (z == 0){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr_s[tz+1][ty][tx] - arr_s[tz][ty][tx]) / dx;
        } else if (z == (nz - 1)){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr_s[tz][ty][tx] - arr_s[tz-1][ty][tx]) / dx;
        } else {
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr_s[tz+1][ty][tx] - arr_s[tz-1][ty][tx]) / (2.0 * dx);
        }
    }
}

/*
 * calc_grad_global_3d :
 *   - 共有メモリを使わず、全ての差分を直接グローバルメモリから読み取って計算する。
 *   - shared版と同等の処理を持つが、性能比較（メモリアクセス最適化の効果測定）に用いられる。
 */
__global__ void calc_grad_global_3d(int nx, int ny, int nz, float dx, float *arr_grad, float *arr){
    // スレッドが担当する全体座標
    const int x = threadIdx.x + blockDim.x * blockIdx.x;
    const int y = threadIdx.y + blockDim.y * blockIdx.y;
    const int z = threadIdx.z + blockDim.z * blockIdx.z;

    // 全体要素数を定義
    const int nxyz = nx * ny * nz;

    // 線形インデックス
    int ijk = nx * ny * z + nx * y + x;

    if (x < nx && y < ny && z < nz){
        int ijk_f;
        int ijk_b;

        // ------------------------------
        // X方向勾配
        // ------------------------------
        ijk_f = nx * ny * z + nx * y + (x + 1);
        ijk_b = nx * ny * z + nx * y + (x - 1);
        if (x == 0){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (x == (nx - 1)){
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            arr_grad[nxyz * X_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }

        // ------------------------------
        // Y方向勾配
        // ------------------------------
        ijk_f = nx * ny * z + nx * (y + 1) + x;
        ijk_b = nx * ny * z + nx * (y - 1) + x;
        if (y == 0){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (y == (ny - 1)){
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            arr_grad[nxyz * Y_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }

        // ------------------------------
        // Z方向勾配
        // ------------------------------
        ijk_f = nx * ny * (z + 1) + nx * y + x;
        ijk_b = nx * ny * (z - 1) + nx * y + x;
        if (z == 0){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk]) / dx;
        } else if (z == (nz - 1)){
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr[ijk] - arr[ijk_b]) / dx;
        } else {
            arr_grad[nxyz * Z_DIRECTION + ijk] = (arr[ijk_f] - arr[ijk_b]) / (2.0 * dx);
        }
    }
}

/*
 * 実験的意義：
 *   - calc_grad_shared_3d は、隣接データを共有メモリにキャッシュし、アクセスパターンを局所化する。
 *     これによりメモリアクセス帯域の削減とスループット向上を狙う。
 *   - calc_grad_global_3d は、最も単純なアクセス（常にグローバルメモリ）を行う基準実装。
 *     両者の実行時間を比較することで shared memory の有効性を定量的に評価できる。
 */