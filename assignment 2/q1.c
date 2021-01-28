#include <stdio.h>
#include <time.h>
#include <stdlib.h>

void addVec(int* C, int* A, int* B, int size);

int main(){
    int size = 50000000;
    clock_t start, end;

    int *A = (int *)calloc(size, sizeof(int));
    int *B = (int *)calloc(size, sizeof(int));
    int *C = (int *)calloc(size, sizeof(int));

    if (A==NULL || B==NULL || C==NULL)
    {
        fprintf(stderr, "Not enough memory\n");
        exit(EXIT_FAILURE);
    }
    
    start = clock();
    addVec(C, A, B, size);
    end = clock();

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", C[i]);
    }

    double exeTime = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
    printf("Execution time: %f ms\n", exeTime);
    

    return 0;
}

void addVec(int* C, int* A, int* B, int size){
    while (size > 0)
    {   *C++ = *A++ + *B++;
        size--;
    }
}

// output:
// 0 0 0 0 0 0 0 0 0 0 Execution time: 132.085000 ms