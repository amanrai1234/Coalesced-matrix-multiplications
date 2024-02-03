
/*
Note: This is colaseinglased matrix multiplication in GPU's 

1]smaller threads blcoks for finer memeory acces patterns,  think of cahche hit and miss in terms of 
all threads row gets laoded when one leemnt is called, yeah we can avoid this by having a size lets say [dim3 blockDim(16,16) blocks in x and y dimeneiosn,i.e
2]SM's are directly connected to the blocks, like A100 has 108 SM's and each SM can deal with around 1024 thread blocks that is massive
3]Specifying a block size that is ideal like 256 elements or 16,16 is a good way to ensure memory colaseing
4]Grid is basically the total size of the problem here (*****total number of blocks==grid) , we have 16,16 elements in x and y directions so the ize of the grid would be 
either it is N/16 or (N*N)/16*16;  so finally dim3 gridDim(N/16,N/16);

*/


#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include<stdlib.h>
#define N 256

// Function for CPU-based matrix multiplication (for verification)
void simple(int* a, int* b, int* c) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                c[i * N + j] += a[i * N + k] * b[k * N + j];
            }
        }
    }
}

// Function to verify the GPU result against the CPU result
void verify(int* c1, int* c2) {
    for (int i = 0; i < N * N; i++) {
        assert(c1[i] == c2[i]);
    }
}

// Function to assign values to matrices a, b, and c
void assign_values(int* a, int* b, int* c) {
    for (int i = 0; i < N * N; i++) {
        a[i] = 1;
        b[i] = 1;
        c[i] = 0; // Initialize c to zero
    }
}

// CUDA kernel for matrix multiplication
__global__ void matmul(int* a, int* b, int* c) {
     int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
     if (row < N && col < N) {
         int sum = 0;
          for (int i = 0; i < N; i++) {
          c[row * N + col] += a[row * N + i] * b[i * N + col];
        }
           
    }
}

int main() {
cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

// Allocate memory for matrices on CPU
int* a = (int*)malloc(sizeof(int) * N * N);
int* b = (int*)malloc(sizeof(int) * N * N);
int* c = (int*)malloc(sizeof(int) * N * N);

// Assign values to matrices a and b
assign_values(a, b, c);

// Allocate memory for matrices on GPU
int* a1, * b1, * c1;
cudaMalloc((void**)&a1, sizeof(int) * N * N);
cudaMalloc((void**)&b1, sizeof(int) * N * N);
cudaMalloc((void**)&c1, sizeof(int) * N * N);

// Copy matrices from CPU to GPU
cudaMemcpy(a1, a, sizeof(int) * N * N, cudaMemcpyHostToDevice);
cudaMemcpy(b1, b, sizeof(int) * N * N, cudaMemcpyHostToDevice);
cudaMemcpy(c1, c, sizeof(int) * N * N, cudaMemcpyHostToDevice);

// Launch the CUDA kernel
dim3 blockDim(16, 16);
dim3 gridDim(N/16, N/16);

cudaEventRecord(start);
matmul<<<gridDim, blockDim>>>(a1, b1, c1);
cudaEventRecord(stop);

// Synchronize and calculate elapsed time
cudaDeviceSynchronize();
float milliseconds = 0.0f;
cudaEventElapsedTime(&milliseconds, start, stop);

// Copy the result matrix from GPU to CPU
cudaMemcpy(c, c1, sizeof(int) * N * N, cudaMemcpyDeviceToHost);

// Verify the result against the CPU-based computation
int* com1 = (int*)malloc(sizeof(int) * N * N);
simple(a, b, com1);
verify(com1, c);

// Free memory on CPU and GPU
free(com1);
free(a);
free(b);
free(c);
cudaFree(a1);
cudaFree(b1);
cudaFree(c1);

// Destroy CUDA events
cudaEventDestroy(start);
cudaEventDestroy(stop);

return 0;
}