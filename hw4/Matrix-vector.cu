#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>

void product_cpu(double* sum_ptr, const double* a, const double* b, long N){
  double sum = 0;
  #pragma omp parallel for schedule(static) reduction(+:sum)
  for (long i = 0; i < N; i++) sum += a[i]*b[i];
  *sum_ptr = sum;
}


#define BLOCK_SIZE 32

__global__ void product(double *A, double *B, double *C, long N) {
    int row = (blockIdx.y) * blockDim.y + threadIdx.y;
    int col = (blockIdx.x) * blockDim.x + threadIdx.x;
    double sum = 0.0;
    if (row < N && col < N) {
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * N + col];
        }
    }
    C[row * N + col] = sum;
}

void Check_CUDA_Error(const char *message) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "ERROR: %s: %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}



__global__ void product_0(double *a, double *b, double *c, long N) {
    int idx = (blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] * b[idx];
}


int main() {
    long N = (1UL << 16);
    double *A, *B, *C, *C_ref;
    cudaMallocHost((void **) &A, N * N * sizeof(double));
    cudaMallocHost((void **) &B, N * N * sizeof(double));
    cudaMallocHost((void **) &C, N * N * sizeof(double));
    cudaMallocHost((void **) &C_ref, N * N * sizeof(double));

#pragma omp parallel for schedule(static) collapse(2)
    for (long i = 0; i < N; i++) {
        for (long j = 0; j < N; j++) {
            A[i * N + j] = 1.0 / (i + 1);
            B[i * N + j] = 1.0 / (i + 1);
        }
    }
    double tt = omp_get_wtime();
    product_cpu(C_ref, A, B, N);
    printf("CPU Bandwidth = %f GB/s\n", N * N * sizeof(double) / (omp_get_wtime() - tt) / 1e9);

    double *A_cuda, *B_cuda, *C_cuda;
    cudaMalloc(&A_cuda, N * N * sizeof(double));
    cudaMalloc(&B_cuda, N * N * sizeof(double));
    cudaMalloc(&C_cuda, N * N * sizeof(double));

    cudaMemcpyAsync(A_cuda, A, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(B_cuda, B, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    
    tt = omp_get_wtime();

    long Nb = (N + BLOCK_SIZE-1) / (BLOCK_SIZE);
    dim3 Blocks(BLOCK_SIZE, BLOCK_SIZE);
    dim3 Grids(Nb, Nb);
    product << < Grids, Blocks >> > (C_cuda, A_cuda, B_cuda, N);

    // product <<<Nb,BLOCK_SIZE>>>(C_cuda, A_cuda, B_cuda, N);
    cudaMemcpyAsync(C, C_cuda, N * N * sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    printf("GPU Bandwidth = %f GB/s\n", N * N * sizeof(double) / (omp_get_wtime() - tt) / 1e9);
    
    double sumdiff = 0.0f;
    for (int row = 0; row < N; row++) {
        for (int col = 0; col < N; col++) {
            double diff = fabs(C[row * N + col] - C_ref[row * N + col]);
            sumdiff += diff;
        }
    }
    printf("Error = %f\n", sumdiff);

    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
    cudaFreeHost(C_ref);
    cudaFree(A_cuda);
    cudaFree(B_cuda);
    cudaFree(C_cuda);

    return 0;
}