#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h> // Corrected header file name
#include <cassert> // For assert

#define S 1024
#define N 1024

__global__ void matmul(int *a, int *b, int *c){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sums = 0;
    if(row < N && col < N){
        for(int k = 0; k < N; k++){
            sums += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sums;
    }
}

// Generate function needs to be defined outside of main
void generate(int* a, int* b, int* ver) {
    for(int i = 0; i < N*N; i++){ // Fixed loop to iterate through entire array
        a[i] = 1;
        b[i] = 1;
        ver[i] = 1;
    }
}

void simple_multiplication(int* a, int* b, int* ver) {
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            for(int k = 0; k < N; k++){ // Corrected K to N
                ver[j * N + i] += a[j * N + k] * b[k * N + i];
            }
        }
    }
}

void verify(int* ver, int* c) {
    for(int i = 0; i < N*N; i++){ // Ensure verification checks the entire array
        assert(ver[i] == c[i]);
    }
}

int main(){
    // Corrected cudaEvent_t for event record declaration
    cudaEvent_t start, stop;

    // Allocate memory on CPU
    int* a = (int*)malloc(sizeof(int) * N * N);
    int* b = (int*)malloc(sizeof(int) * N * N);
    int* c = (int*)malloc(sizeof(int) * N * N);
    int* ver = (int*)malloc(sizeof(int) * N * N);

    // Generate initial values
    generate(a, b, ver);

    // Allocate memory on GPU
    int* a1;
    int* b1;
    int* c1;

    cudaMalloc((void**)&a1, sizeof(int) * N * N);
    cudaMalloc((void**)&b1, sizeof(int) * N * N);
    cudaMalloc((void**)&c1, sizeof(int) * N * N);

    // Copy from host to device
    cudaMemcpy(a1, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(b1, b, sizeof(int) * N * N, cudaMemcpyHostToDevice); // Corrected direction

    // Specify block and grid parameters
    dim3 blockDim(32, 32);
    dim3 gridDim((N + 31) / 32, (N + 31) / 32); // Ensure enough blocks to cover all data

    // Start timer
    cudaEventCreate(&start);
    cudaEventRecord(start);

    // Launch the kernel
    matmul<<<gridDim, blockDim>>>(a1, b1, c1);

    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Elapsed time: " << milliseconds << " ms\n";

    // Copy result back to host
    cudaMemcpy(c, c1, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

    // Verify the result
    simple_multiplication(a, b, ver);
    verify(ver, c);

    // Free GPU memory
    cudaFree(a1);
    cudaFree(b1);
    cudaFree(c1);

    // Free CPU memory
    free(a);
    free(b);
    free(c);
    free(ver);

    return 0;
}
