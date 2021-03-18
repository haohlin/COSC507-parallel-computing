#include <stdio.h>
#include <stdlib.h>
//#include "cuda_runtime.h"

__global__ void init(double* d_a, const int n);

int main(){
    const int n = 10000000;
    double* a = (double *)malloc(n * sizeof(double));
    double* d_a = 0;
    int blockSize = 1024, gridSize =  (n-1)/blockSize + 1;
    // float time = 0;
    cudaMalloc(&d_a, n * sizeof(double));

    double t = clock();
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    init<<<gridSize,blockSize>>>(d_a, n);
    // cudaEventRecord(stop); 
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&time, start, stop); //time in milliseconds
    // cudaEventDestroy(start);
    // cudaEventDestroy(stop);
    cudaDeviceSynchronize();
    t = 1000 * (clock()-t) / CLOCKS_PER_SEC;

    cudaMemcpy(a, d_a, n * sizeof(double), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 5; i++){
        printf("a[%d]: %.7f\n", i, a[i]);
    }
    printf("...\n");
    for(int i = 5; i > 0; i--){
        printf("a[%d]: %.7f\n", n-i, a[n-i]);
    }
    printf("GPU time: %f\n", t);

    free(a);
    cudaFree(d_a);
    return 0;
}

__global__ void init(double* d_a, const int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        d_a[i] = (double)i / n;
}

//GPU time: 0.36ms