#include <stdio.h>

// 2つの整数を加算して結果を返す関数である
int add_two_number(int num1, int num2){
    int res;            // 加算結果を格納するための整数変数を宣言
    res = num1 + num2;  // num1とnum2を加算し、その結果をresに代入
    return res;         // 計算結果を呼び出し元へ返す
}

int main(){
    int num1 = 1;
    int num2 = 2;
    int result = add_two_number(num1, num2);
    printf("result = %d\n", result);
    return 0;
}