#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define N 10

int main(){
    int *a = (int*)malloc(sizeof(int) * N);
    int *a_copy = (int*)calloc(sizeof(int), N);

    a[0] = 0; 

    # pragma omp parallel for 
    for(int i=1; i<N; i++){
        a[i] = i * (i + 1) / 2; 
    } 

    for(int i=0; i<N; i++){
        printf("%d, ", a[i]); 
    }

    return 0;
}

/*
Output: 0, 1, 3, 6, 10, 15, 21, 28, 36, 45

a[i] = 1 + 2 + ... + i = i * (i + 1) / 2
*/

