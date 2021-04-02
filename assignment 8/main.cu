#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include "device_launch_parameters.h"
#define CHK(call) {												\
	cudaError_t err = call;										\
	if (err != cudaSuccess){									\
		printf("Error%d: %s: %d\n", err, __FILE__, __LINE__);	\
	printf(cudaGetErrorString(err));							\
	cudaDeviceReset();											\
		exit(1);												\
	}															\
}
void checkForGPU() {
    // This code attempts to check if a GPU has been allocated.
    // Colab notebooks without a GPU technically have access to NVCC and will
    // compile and execute CPU/Host code, however, GPU/Device code will silently
    // fail. To prevent such situations, this code will warn the user.
    int count;
    cudaGetDeviceCount(&count);
    if (count <= 0 || count > 100) {
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
        printf("->WARNING<-: NO GPU DETECTED ON THIS COLLABORATE INSTANCE.\n");
        printf("IF YOU ARE ATTEMPTING TO RUN GPU-BASED CUDA CODE, YOU SHOULD CHANGE THE RUNTIME TYPE!\n");
        printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n");
    }
}

__global__ void cuda_sum(double *d_arr, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        for(int stride = 1; stride < blockDim.x; stride *= 2){
            if(i % (2 * stride) == 0)
                d_arr[i] += d_arr[i+stride];
            __syncthreads();
        }
    }
}

__global__ void cuda_sum_bin(double *d_arr, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        for(int stride = 1; stride < blockDim.x; stride <<= 1){
            if(i % (stride << 1) == 0)
                d_arr[i] += d_arr[i+stride];
            __syncthreads();
        }
    }
}

__global__ void cuda_sum_shared(double *d_arr, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ double temp_arr[1024];
    if(i < N){
        temp_arr[tid] = d_arr[i];
        __syncthreads();
        for(int stride = 1; stride < blockDim.x; stride *= 2){
            if(i % (2 * stride) == 0)
                temp_arr[tid] += temp_arr[tid+stride];
            __syncthreads();
        }
        if(tid == 0)
            d_arr[i] = temp_arr[tid];
    }
}

__global__ void cuda_sum_no_div(double *d_arr, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    if(i < N){
        for(int stride = blockDim.x/2; stride >= 1; stride = stride / 2){
            if(tid < stride)
                d_arr[i] += d_arr[i+stride];
            __syncthreads();
        }
    }
}

__global__ void cuda_sum_shared_no_div(double *d_arr, int N){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;
    __shared__ double temp_arr[1024];
    if(i < N){
        temp_arr[tid] = d_arr[i];
        __syncthreads();
        for(int stride = blockDim.x/2; stride >= 1; stride /= 2){
            if(tid < stride)
                temp_arr[tid] += temp_arr[tid+stride];
            __syncthreads();
        }
        if(tid == 0)
            d_arr[i] = temp_arr[tid];
    }
}

int main() {
    checkForGPU();

    // Your implementation goes here.
    // It's recommended that you keep the checkForGPU code since it serves
    // as a warning against silent CUDA failures on Colab.

    int N = pow(2,24), nthreads = 1024;
    int n_blocks = 1 + (N - 1) / nthreads;
    size_t arrSize = N * sizeof(double);
    double sum = 0.0, sum_init = 0.0, t;
    double *arr = (double*)malloc(arrSize);
    double *init_arr = (double*)malloc(arrSize);
    double *d_arr = NULL;
    //srand((unsigned int)time(NULL));
    // Init
    for(int i = 0; i < N; i++){
        init_arr[i] = ((double)rand()/(double)(RAND_MAX)) * 255.0;//
    }

    printf("Reducing an array of %d floats on a grid of (%d, %d, %d) blocks, each bloack with (%d, %d, %d) threads\n\n",\
            N, n_blocks,1,1, nthreads,1,1);
    CHK(cudaMalloc(&d_arr, arrSize));

    t = clock();
    CHK(cudaMemcpy(d_arr, init_arr, arrSize, cudaMemcpyHostToDevice));
    cuda_sum<<<n_blocks,nthreads>>>(d_arr, N);
    CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    CHK(cudaMemcpy(arr, d_arr, arrSize, cudaMemcpyDeviceToHost);)
    sum = 0.0;
    for(int i = 0; i < N; i += nthreads){
        sum += arr[i];
    }
    printf("Using global memory, More divergence: GPU time: %fms GPU sum: %f\n", t, sum);
    
    t = clock();
    CHK(cudaMemcpy(d_arr, init_arr, arrSize, cudaMemcpyHostToDevice));
    cuda_sum_bin<<<n_blocks,nthreads>>>(d_arr, N);
    CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    CHK(cudaMemcpy(arr, d_arr, arrSize, cudaMemcpyDeviceToHost));
    sum = 0.0;
    for(int i = 0; i < N; i += nthreads){
        sum += arr[i];
    }
    printf("Using global memory, More divergence (with binary operation): GPU time: %fms GPU sum: %f\n", t, sum);

    t = clock();
    CHK(cudaMemcpy(d_arr, init_arr, arrSize, cudaMemcpyHostToDevice));
    cuda_sum_shared<<<n_blocks,nthreads>>>(d_arr, N);
    CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    cudaMemcpy(arr, d_arr, arrSize, cudaMemcpyDeviceToHost);
    sum = 0.0;
    for(int i = 0; i < N; i += nthreads){
        sum += arr[i];
    }
    printf("Using shared memory, More divergence: GPU time: %fms GPU sum: %f\n", t, sum);

    t = clock();
    CHK(cudaMemcpy(d_arr, init_arr, arrSize, cudaMemcpyHostToDevice));
    cuda_sum_no_div<<<n_blocks,nthreads>>>(d_arr, N);
    CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    CHK(cudaMemcpy(arr, d_arr, arrSize, cudaMemcpyDeviceToHost));
    sum = 0.0;
    for(int i = 0; i < N; i += nthreads){
        sum += arr[i];
    }
    printf("Using global memory, Less divergence: GPU time: %fms GPU sum: %f\n", t, sum);

    t = clock();
    CHK(cudaMemcpy(d_arr, init_arr, arrSize, cudaMemcpyHostToDevice));
    cuda_sum_shared_no_div<<<n_blocks,nthreads>>>(d_arr, N);
    CHK(cudaGetLastError());
	CHK(cudaDeviceSynchronize());
    t = (clock() - t) * 1000 / CLOCKS_PER_SEC;
    CHK(cudaMemcpy(arr, d_arr, arrSize, cudaMemcpyDeviceToHost));
    sum = 0.0;
    for(int i = 0; i < N; i += nthreads){
        sum += arr[i];
    }
    printf("Using shared memory, Less divergence: GPU time: %fms GPU sum: %f\n", t, sum);

    // cuda_sum_shared<<<n_blocks,nthreads>>>(d_arr, N);
    //cuda_sum_no_div<<<n_blocks,nthreads>>>(d_arr, N);
    //cuda_sum_shared_no_div<<<n_blocks,nthreads>>>(d_arr, N);



    for(int i = 0; i < N; i++){
        sum_init += init_arr[i];
        // printf("arr[i]: %f\n", arr[i]);
    }
    printf("ground truth: %f\n", sum_init);
    free(arr);
    cudaFree(d_arr);
    return 0;
}

/*
Results not accurate due to floating point imprecision.
Reducing an array of 16777216 floats on a grid of (16384, 1, 1) blocks, each bloack with (1024, 1, 1) threads

Using global memory, More divergence: GPU time: 11.497000ms GPU sum: 2139216579.235488
Using global memory, More divergence (with binary operation): GPU time: 11.830000ms GPU sum: 2139216579.235488
Using shared memory, More divergence: GPU time: 11.881000ms GPU sum: 2139216579.235488
Using global memory, Less divergence: GPU time: 11.006000ms GPU sum: 2139216579.235488
Using shared memory, Less divergence: GPU time: 11.562000ms GPU sum: 2139216579.235488
*/