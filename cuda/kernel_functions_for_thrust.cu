#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>

extern "C" {

/*
 * thrust::sort を用いて整数配列を昇順ソートしようとする関数です。
 *
 * 【重要な注意】
 * - thrust のアルゴリズム（thrust::sort, thrust::sort_by_key など）は
 *   基本的に「ホスト関数」から起動する設計です。つまり __global__ 関数（デバイスカーネル）
 *   の本体の中で直接呼ぶことは想定されていません（コンパイルエラーや未定義動作の原因になります）。
 * - thrust の実行ポリシー thrust::device は「デバイスで実行する」ことを指示しますが、
 *   それでも起動地点は“ホスト側”である必要があります（ホストから thrust::sort(thrust::device, ...) を呼ぶ）。
 *
 * 【期待される前提（ホストから呼ぶ場合）】
 * - arr はデバイスメモリ上の先頭ポインタ（int*、cudaMalloc 等で確保済み）であること。
 * - num_component は要素数です。範囲 [arr, arr + num_component) をソート対象にします。
 * - thrust を使う正しい方法は、ホストコード側で
 *     thrust::sort(thrust::device, arr, arr + num_component);
 *   を呼び出す形です（__global__ は不要）。
 *
 * 【extern "C" について】
 * - シンボル名の C リンク指定です。C から呼び出す場合や PyCUDA/FFI などで名前修飾を避けたいときに有用です。
 *   ただし本コードのように __global__ にした上で thrust を内部で呼ぶのは避けてください。
 */
__host__ void sort_thrust(int num_component, int *arr){
    // ここで thrust::sort を呼ぶのは推奨されません（通常はホスト側から呼びます）。
    // 正しい使用例（ホスト側）:
    //   thrust::sort(thrust::device, arr, arr + num_component);
    thrust::sort(arr, arr + num_component);
}

/*
 * thrust::sort_by_key を用いてキー配列 key に基づき、value を連動させて並べ替える関数です。
 *
 * 【使いどころ】
 * - key: ソートの基準となる整数配列（デバイスメモリ上）
 * - value: key と同じ長さの付随値配列（デバイスメモリ上）
 * - 結果として key は昇順に並び、value は対応関係を保ったまま同じ並び替えが適用されます。
 *
 * 【重要な注意（上と同様）】
 * - thrust のアルゴリズム呼び出しはホスト側で行うのが前提です。
 *   __global__ の中に直接書くのではなく、ホストコード側で
 *     thrust::sort_by_key(thrust::device, key, key + num_component, value);
 *   のように呼び出してください。
 *
 * 【性能メモ】
 * - thrust は内部で効率的な並列ソート（Radix Sort など）へディスパッチされるため、
 *   手書きの汎用ソートより高速なことが多いです。
 * - メモリアクセスは連続領域を想定しています。アライメントやピッチ付き配列の場合は注意してください。
 */
__host__ void sort_by_key_thrust( int num_component, int *key, int *value){
    // ここで thrust::sort_by_key を呼ぶのは推奨されません（通常はホスト側から呼びます）。
    // 正しい使用例（ホスト側）:
    //   thrust::sort_by_key(thrust::device, key, key + num_component, value);
    thrust::sort_by_key(key, key + num_component, value);
}

}