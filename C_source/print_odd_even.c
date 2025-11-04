#include <stdio.h>

void print_odd_even(int num){
    // 0 以上 num 未満の整数 i について奇数・偶数判定を行い標準出力へ表示する
    // for ループの初期化: i を 0 に設定し、条件 i < num を満たす間インクリメントする
    for (int i = 0; i < num; i++){
        // i を 2 で割った余りが 0 であれば偶数である
        if (i % 2 == 0){
            printf("%d is an even number.\n", i);
        // 上の条件の否定（i % 2 == 0 ではない）すなわち奇数である
        } else if (!(i % 2 == 0)){
	        printf("%d is an odd number.\n", i);
        // 上記のいずれにも当てはまらない場合（理論上到達しない）は異常メッセージを出す
        } else {
            printf("Something wrong...\n");
        }
    }
}

int main(){
    int num = 10;
    print_odd_even(num);
    return 0;
}