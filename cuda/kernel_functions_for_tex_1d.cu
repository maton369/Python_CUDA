/*
 * 1次元テクスチャメモリから値を読み取る CUDA カーネルの例である。
 *
 * 背景と要点：
 * - `texture<int, 1, cudaReadModeElementType>` は **レガシー（旧API）のテクスチャ参照**であり、
 *   カーネル起動前にホスト側で `cudaBindTexture` などを用いて線形メモリや CUDA 配列にバインドする必要がある。
 * - `cudaReadModeElementType` は、メモリ中の値をそのままの型（ここでは `int`）として読み出すモードである。
 * - テクスチャ読み出しは `tex1Dfetch(tex_1d, index)` を用いる。1D の場合、整数添字でランダムアクセスができる。
 * - テクスチャメモリは専用のキャッシュを持ち、アクセスパターンによってはグローバルメモリより有利になる場合がある。
 *   （ただし近年は L1/L2 キャッシュが強化されており、常に有利とは限らない。）
 *
 * 使い方の概略（ホスト側で必要な準備の例）：
 *   1) `cudaChannelFormatDesc desc = cudaCreateChannelDesc<int>();`
 *   2) `cudaBindTexture(NULL, tex_1d, dev_ptr, desc, nx*sizeof(int));`
 *   3) 本カーネルを適切な block/grid で起動する。
 *   4) 使用後 `cudaUnbindTexture(tex_1d);` でアンバインドする。
 *
 * 注意：
 * - 本コードは **テクスチャ参照（旧API）**の記法であり、**テクスチャオブジェクト（近年の推奨API）**とは異なる。
 * - `printf` はデバイス側の標準出力に出力するため、デバッグ時に有用であるがオーバーヘッドが大きい。
 *   大規模スレッドでの大量出力はタイムアウトやログ肥大を招くため注意が必要である。
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <texture_types.h>
#include <stdio.h>

// CUDA 12.0以降ではレガシーテクスチャ参照が削除されているため、
// 互換性のために手動で定義を追加
// 注意: これは互換性のための一時的な解決策です。
// 可能であればCUDA 11.x以前を使用するか、新しいテクスチャオブジェクトAPIに移行してください。

// texture テンプレートの互換性定義を別ヘッダーからインクルード
#if CUDA_VERSION >= 12000
#include "texture_compat.h"
#endif

// レガシーな 1D テクスチャ参照を宣言するである。
// 第1テンプレート引数: 要素型（int）
// 第2テンプレート引数: 次元数（1）
// 第3テンプレート引数: 読み出しモード（要素型そのまま）である。
#if CUDA_VERSION >= 12000
// CUDA 12.0以降では、レガシーテクスチャ参照APIが削除されているため、
// PyCUDAのget_texrefが動作しない可能性があります。
// グローバルスコープで定義し、シンボルとしてエクスポート可能にする
__device__ texture<int, 1, cudaReadModeElementType> tex_1d = {0};
#else
texture<int, 1, cudaReadModeElementType> tex_1d;
#endif

__global__ void read_texture_1d(int nx){
   // スレッドのグローバルな 1D インデックスを計算するである。
   // blockDim.x: ブロック内のスレッド数（x次元）
   // blockIdx.x: グリッド内のブロック番号（x次元）
   // threadIdx.x: ブロック内のスレッド番号（x次元）
   int x = threadIdx.x + blockDim.x * blockIdx.x;

   // 範囲外アクセスを防ぐである。テクスチャ参照先（バインドした領域）の長さは nx 要素を想定する。
   if (x < nx){
        // 1次元テクスチャから x 番目の要素を読み出すである。
        // tex1Dfetch は整数添字による直接アクセスを提供し、非正規化座標を使用する。
        int value = tex1Dfetch(tex_1d, x);

        // デバッグ出力である。各スレッドが自分のIDと読み出した値を表示する。
        // 注意：大量スレッドでの printf はパフォーマンスへ影響が大きい。
        printf("my id is %d, my value is %d\n", x, value);
   }
}