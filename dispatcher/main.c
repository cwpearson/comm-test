#include "dispatcher.h"
#include <stdlib.h>
int main(void)
{
    int *x = (int *)malloc(sizeof(int));
    *x = 5;
    int y = giveToDispatch(x, sizeof(x), 0);

    // for (int j=0; j<500000; j+=2){
    //     j--;
    // }

    int *a = (int *)getFromDispatch(0);
    int b = *a;
    printf("Integer to print is %d \n", b);
    return 0;
}
