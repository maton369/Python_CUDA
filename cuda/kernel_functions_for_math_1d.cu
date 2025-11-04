/*
 * CUDAグローバル関数（カーネル）である。
 * 目的: 入力配列 x の各要素に 1 を加算し、結果を出力配列 y に書き込む。
 * 実行モデル: 多数のスレッドが並列に同一処理を行い、配列の異なるインデックスを担当する。
 *
 * 引数:
 *   num_comp : 配列の要素数（境界チェックに使用）
 *   y        : 出力先デバイス配列（int*）
 *   x        : 入力元デバイス配列（int*）
 *
 * メモ:
 *   - 本カーネルは1次元のスレッド構成を前提とした典型的なインデックス計算を用いている。
 *   - スレッドあたり 1 要素を処理する要素並列（element-wise）パターンである。
 */
__global__ void plus_one_kernel(int num_comp, int *y, int *x){
    // グローバルインデックス i を計算するである。
    // threadIdx.x  : ブロック内スレッドID（0..blockDim.x-1）
    // blockDim.x   : ブロック内のスレッド総数
    // blockIdx.x   : グリッド内ブロックID（0..gridDim.x-1）
    // よって i は「全体の中でこのスレッドが担当する要素番号」を表す。
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // 範囲外アクセス防止のための境界チェックである。
    // blocks_per_grid * threads_per_block が num_comp を超える（過剰スレッドを起動）構成でも安全にする。
    if (i < num_comp){
        // 各スレッドが自分の担当要素を 1 増加させて書き戻すである。
        // 連続インデックスに対し連続アドレスへアクセスするため、warp単位で概ねcoalesced access が期待できる。
        y[i] = x[i] + 1;
    }
}