#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(){
    int len, sum = 0;
    clock_t start, end;
    double exeTime;
    printf("Input vector size: ");
    scanf("%d", &len);

    int *A = (int *)malloc(len * sizeof(int));
    int *B = (int *)malloc(len * sizeof(int));
    int *C = (int *)malloc(len * sizeof(int));

    if (A==NULL || B==NULL || C==NULL)
    {
        printf("Not enough memory.\n");
        exit(0);
    }

    start = clock();
    for (int i = 0; i < len; i++)
    {
        A[i] = 3 * i;
        B[i] = - 3 * i;
        C[i] = A[i] + B[i];
        sum += C[i];
    }
    end = clock();
    exeTime = (double) (end - start) / CLOCKS_PER_SEC;
    printf("Sum: %d\nExecution time: %.2f", sum, exeTime);
    free(A);
    free(B);
    free(C);
    return 0;
}

/*
Vector size        Sum          Execution time
1m                 0            0.01s
10m                0            0.08s
50m                0            0.38s
100m               0            0.76s
*/