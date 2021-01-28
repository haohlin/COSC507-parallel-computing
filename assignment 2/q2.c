#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int *addVec2(int* A, int* B, int size);

int main(){
    int size = 50000000;
    clock_t start, end;

    int *A = (int *)calloc(size, sizeof(int));
    int *B = (int *)calloc(size, sizeof(int));
    int *C = NULL;

    
    
    start = clock();
    C = addVec2(A, B, size);
    end = clock();

    if (C==NULL)
    {
        fprintf(stderr, "Not enough memory\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < 10; i++)
    {
        printf("%d ", C[i]);
    }

    double exeTime = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
    printf("Execution time: %f ms\n", exeTime);
    

    return 0;
}

int *addVec2(int* A, int* B, int size){
    int *C = (int *)calloc(size, sizeof(int));
    for (int i = 0; i < size; i++)
    {
        C[i] = A[i] + B[i];
    }
    return C;
}

// output:
// 0 0 0 0 0 0 0 0 0 0 Execution time: 271.266000 ms