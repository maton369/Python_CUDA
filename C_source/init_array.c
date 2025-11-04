#include <stdio.h>

void init_array(int num_comp, float *arr){
    // 配列 arr の各要素を 0.0 で初期化する関数である
    // 引数:
    //   num_comp : 配列の要素数（ループ回数の上限）
    //   arr      : 初期化対象となる float 型配列へのポインタ
    // 注意:
    //   - arr が有効なメモリ領域を指していることを前提としている。
    //   - num_comp が負の値である場合、for ループは実行されない。
    //   - 要素数より大きい値を渡すとメモリ破壊（オーバーラン）を起こすため注意が必要である。
    for (int i = 0; i < num_comp; i++){
        arr[i] = 0.0;  // 各要素を 0.0（float のゼロ）で初期化
    }
}

int main(){
    int num_comp = 10;
    float arr[num_comp];
    init_array(num_comp, arr);
    for (int i = 0; i < num_comp; i++){
        printf("arr[%d] = %f\n", i, arr[i]);
    }
    return 0;
}