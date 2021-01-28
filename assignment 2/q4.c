#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void test(int *A, int size, clock_t start, clock_t end, int mode, int num_thread);
int* vecCreate (int size);
int* vecCreateOpenMP(int size, int num_thread);

int main(){
    int num_thread, size = 50000000, *A = NULL;
    int mode;
    clock_t start, end;

    printf("Enter number of threads: ");
    scanf("%d", &num_thread);

    start = clock();
    A = vecCreate(size);
    end = clock();
    test(A, size, start, end, mode=1, num_thread);

    if (size%num_thread != 0)
    {
        fprintf(stderr, "Error: number of threads must be divisible by vector size.\n");
        exit(EXIT_FAILURE);
    }

    start = clock();
    A = vecCreateOpenMP(size, num_thread);
    end = clock();
    test(A, size, start, end, mode=0, num_thread);
    
    return 0;
}

void test(int *A, int size, clock_t start, clock_t end, int mode, int num_thread){
    if (A==NULL)
    {
        fprintf(stderr, "Not enough memory\n");
        exit(EXIT_FAILURE);
    }
    if (mode==1) printf("Using serial code\n");
    else printf("Using OpenMP with %d threads\n", num_thread);

    printf("v[%d] = %d\n", size - 1, A[size - 1]);
    double time = (double)(end - start) * 1000 / CLOCKS_PER_SEC;
    printf("Time: %f ms\n\n", time);
}

int* vecCreate (int size){
    int *A = (int*)malloc(size * sizeof(int));
    for (int i = 0; i < size; i++)
    {
        A[i] = i;
    }
    return A;
}

int* vecCreateOpenMP(int size, int num_thread){
    int *A = (int*)malloc(size * sizeof(int));
    int seg_len = size / num_thread;
    
    
    #pragma omp parallel num_threads(num_thread)
    {
        int tid = omp_get_thread_num();
        int start = tid * seg_len;
        for (int i = 0; i < seg_len; i++)
        {
            A[start + i] = start + i;
        }
    }

    return A;
}

// Output example 1:
// Enter number of threads: 4
// Using serial code
// v[49999999] = 49999999
// Time: 155.352000 ms

// Using OpenMP with 4 threads
// v[49999999] = 49999999
// Time: 100.248000 ms


// Output example 2:
// Enter number of threads: 20
// Using serial code
// v[49999999] = 49999999
// Time: 134.358000 ms

// Using OpenMP with 20 threads
// v[49999999] = 49999999
// Time: 199.236000 ms
// (Why slower with 20 threads??)


// Output example 3:
// Enter number of threads: 3
// Using serial code
// v[49999999] = 49999999
// Time: 152.849000 ms

// Error: number of threads must be divisible by vector size.