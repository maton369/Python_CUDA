/*
 * 目的:
 * - 任意の型 T に対して「要素同士の加算」を行うデバイス関数と、
 *   1 次元ベクトル a, b の要素ごとの和を res に書き出す CUDA カーネルの最小例。
 *
 * 構成ポイント:
 * - __device__ 関数 add_two_vector<T> はテンプレート化されており、T が加算演算子(+)をサポートしていれば利用可能。
 * - __global__ カーネル add_two_vector_kernel はグリッド上の各スレッドに 1 要素を担当させる 1D マッピング。
 * - スレッドの線形インデックス x を用い、範囲チェック (x < nx) を通過した要素に対して a[x] + b[x] を計算し res[x] に格納。
 * - extern "C" を付けることで名前修飾（C++ のマングリング）を抑制し、ホスト側からシンボルを取得しやすくしている（PyCUDA 等からの呼び出しにも有利）。
 *
 * 実行時の注意:
 * - ホスト側で a, b, res はデバイスメモリ（cudaMalloc 等）上に確保し、必要に応じて Host↔Device 転送を行う。
 * - 起動構成は blockDim.x * gridDim.x >= nx となるように設定し、余剰スレッドは if (x < nx) のガードで無害化。
 * - この例では T=int を使用（add_two_vector<int> を明示インスタンス化）。float/double など他型にも拡張可能。
 * - オーバーフローが懸念される場合は wider type（例: 64bit）や saturating add の検討を。
 */

template <class T>
__device__ T add_two_vector(T x, T y){
    return (x + y);
}

extern "C" {
__global__ void add_two_vector_kernel(int nx, int *a, int *b, int *res){
    // スレッドの 1D グローバルインデックスを算出
    const int x = threadIdx.x + blockDim.x * blockIdx.x;

    // 範囲外アクセスの防止（余剰スレッドを弾く）
    if (x < nx){
        // テンプレートの明示インスタンス化（T=int）を用いて要素同士を加算し、結果を書き込み
        res[x] = add_two_vector<int>(a[x], b[x]);
    }
}
}