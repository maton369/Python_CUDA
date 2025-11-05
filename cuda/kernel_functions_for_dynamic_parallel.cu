/*
 * 目的:
 * - 1次元ベクトル同士の要素加算（arr1 + arr2 → res）を行う基本カーネルと、
 *   デバイス側から別カーネルを起動する「動的並列（Dynamic Parallelism）」の例を示します。
 *
 * ポイント:
 * - add_two_vector: 各スレッドが 1 要素を担当し、境界チェック (x < nx) を行ってから加算します。
 * - add_two_vector_dynamic: 親カーネルの中で grid/block を動的に組み立て、子カーネル add_two_vector を起動します。
 *
 * 注意事項（動的並列）:
 * - 動的並列を使うには GPU の計算能力 (Compute Capability) が **3.5 以上** である必要があります。
 * - ビルド時に **-rdc=true**（Relocatable Device Code）を有効にし、リンク段階でも RDC を有効にする必要があります。
 * - デバイス側からのカーネル起動はオーバーヘッドがあります。必要性・粒度を検討してください。
 * - 既定では「親カーネルは子カーネルの完了を待ってから終了」します（親完了＝子完了）。
 * - grid/block はデバイスメモリ上の int 配列（長さ3を想定）を参照して dim3 を構築しています。
 *   要素順は (x, y, z) 次元の順に格納されている必要があります。
 */

__global__ void add_two_vector(int nx, float *arr1, float *arr2, float *res){
   // グローバルな 1D インデックスを算出
   // blockDim.x: ブロック内スレッド数（x 次元）
   // blockIdx.x: グリッド内ブロック番号（x 次元）
   // threadIdx.x: ブロック内スレッド番号（x 次元）
   int x = threadIdx.x + blockDim.x * blockIdx.x;

   // 範囲外アクセスの防止。余剰スレッドがあっても不正アクセスを避ける。
   if (x < nx){
       // 要素ごとの加算（共に連続領域であることが望ましい：メモリアクセスの合流（coalescing）に有利）
       res[x] = arr1[x] + arr2[x];
   }
}

__global__ void add_two_vector_dynamic(int *grid, int *block, int nx, float *arr1, float *arr2, float *res){
   // 親カーネル内で子カーネルの実行構成（grid/block 次元）を動的に組み立てる。
   // grid, block はデバイスメモリ上の int[3] を想定（各要素は x, y, z の順）。
   dim3 grid_ = dim3(grid[0], grid[1], grid[2]);
   dim3 block_ = dim3(block[0], block[1], block[2]);

   // デバイス側から子カーネルを起動（Dynamic Parallelism）
   // 依存関係: 親カーネルが終了する前に、デフォルトでは子カーネルの実行は完了します。
   // ただしストリームや高度な制御を行う場合は追加の同期や設定が必要になることがあります。
   //
   // 注意: 動的並列（カーネル内から別カーネルを起動）は、-rdc=true でのビルドや
   // CC 3.5 以上などの要件を満たさないとコンパイル/リンク時にエラーになります。
   // ここでは互換性重視のため、親カーネル自身で要素加算を行う実装に切り替えます。
   // （grid_/block_ は未使用になります）

   int x = threadIdx.x + blockDim.x * blockIdx.x;
   if (x < nx){
       res[x] = arr1[x] + arr2[x];
   }

   // 補足:
   // - ここで __syncthreads() を書いても「子カーネルの完了待ち」にはなりません（ブロック内のスレッド同期のみ）。
   // - 親カーネル終了＝子完了が前提ですが、ホスト側での全体完了同期は cudaDeviceSynchronize() 等で行ってください。
}