#include <algorithm>
#include <stdio.h>
#include <omp.h>
#include <string>


void MVM(double* C, const double* A, const double* B, long N) {
  for (long i = 0; i < N; i++) {
    double sum = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum)
    for (long j = 0; j < N; j++) {
      sum += A[i*N+j]*B[j];
    }
    C[i] = sum;
  }
}

#define BLOCK_SIZE 1024

__global__ void reduction(double* sum, const double* a, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = a[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

__global__ void product(double* sum, const double* A, const double* b, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx = (blockIdx.x) * blockDim.x + threadIdx.x;

  if (idx < N) smem[threadIdx.x] = A[idx]*b[idx];
  else smem[threadIdx.x] = 0;

  __syncthreads();
  if (threadIdx.x < 512) smem[threadIdx.x] += smem[threadIdx.x + 512];
  __syncthreads();
  if (threadIdx.x < 256) smem[threadIdx.x] += smem[threadIdx.x + 256];
  __syncthreads();
  if (threadIdx.x < 128) smem[threadIdx.x] += smem[threadIdx.x + 128];
  __syncthreads();
  if (threadIdx.x <  64) smem[threadIdx.x] += smem[threadIdx.x +  64];
  __syncthreads();
  if (threadIdx.x <  32) {
    smem[threadIdx.x] += smem[threadIdx.x +  32];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +  16];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   8];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   4];
    __syncwarp();
    smem[threadIdx.x] += smem[threadIdx.x +   2];
    __syncwarp();
    if (threadIdx.x == 0) sum[blockIdx.x] = smem[0] + smem[1];
  }
}

int main() {
  long N = (1UL<<10);
  double *x, *M;

  cudaMallocHost((void**)&x, N * sizeof(double));
  cudaMallocHost((void**)&M, N*N*sizeof(double));

  #pragma omp parallel for schedule(static)
  for (long i = 0; i < N; i++) {
    x[i] = drand48();
  }
  for (long i = 0; i < N*N; i++) {
    M[i] = drand48();
  }

  double *sum_ref, *sum;
  cudaMallocHost((void**)&sum_ref, N * sizeof(double));
  cudaMallocHost((void**)&sum, N * sizeof(double));

  double tt = omp_get_wtime();
  MVM(sum_ref, M, x, N);
  printf("CPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);
  printf("CPU test %f \n", sum_ref[1]);

  double *x_d, *M_d, *z_d;
  cudaMalloc(&x_d, N*sizeof(double));
  cudaMalloc(&M_d, N*N*sizeof(double));
  cudaMemcpyAsync(x_d, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(M_d, M, N*N*sizeof(double), cudaMemcpyHostToDevice);
    
  long u = 1;
  for (long i = (N+BLOCK_SIZE-1)/(BLOCK_SIZE); i > 1; i = (i+BLOCK_SIZE-1)/(BLOCK_SIZE)){
    u += i;
  }
  cudaMalloc(&z_d, u*sizeof(double)); 


  cudaDeviceSynchronize();

  tt = omp_get_wtime();

  for (long i = 0; i < N; i++) {
    double* sum_d = z_d;
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    product<<<Nb,BLOCK_SIZE>>>(sum_d, M_d+i*N, x_d, N);
    while (Nb > 1) {
      long Nx = Nb;
      Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
      reduction<<<Nb,BLOCK_SIZE>>>(sum_d + Nx, sum_d, Nx);
      sum_d += Nx;
    }
    cudaMemcpyAsync(&sum[i], sum_d, 1*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }

  printf("GPU Bandwidth = %f GB/s\n", N*N*sizeof(double) / (omp_get_wtime()-tt)/1e9);

  double error = 0;
  #pragma omp parallel for reduction(+:error)
  for (long i = 0; i < N; i++) {
    error = error + fabs(sum[i] - sum_ref[i]);
  }
  printf("Error = %f\n", error);
  
  cudaFree(x_d);
  cudaFree(z_d);
  cudaFree(M_d); 
  cudaFreeHost(x);
  cudaFreeHost(M);

  return 0;
}