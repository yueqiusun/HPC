#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
  // for (long i = 1; i < n; i++) {
  //   printf("prefix_sum[%d] = %d\n", i,prefix_sum[i]);
  // }
}

void scan_omp(long* prefix_sum, const long* A, long n, long* part_sum) {
  // TODO: implement multi-threaded OpenMP scan
  int num_threads;
  #pragma omp parallel
  {
    num_threads = omp_get_num_threads();
  }
  printf("Number of threads = %d\n", num_threads);
  int num_k = num_threads; 
  int test = 0;
  if(test == 1){
    num_k = 10;
  }
  
  long piece = n / num_k;
  long start, ind,i,k,end;
  
  prefix_sum[0] = 0;
  part_sum[0] = 0;

  #pragma omp parallel for shared(piece,A,part_sum) private(k,i,start,ind) 
    for (k = 0; k < num_k; k ++){
      start = k*piece + 1;
      part_sum[start] = A[start - 1];
      for (i = 1; i < piece; i++) {
        ind = start + i ;
        part_sum[ind] = part_sum[ind-1] + A[ind-1]; 
      }
    } 
    if (piece * num_k < n-1 ) {
      start = piece * num_k + 1;
      part_sum[start] = A[start - 1];
      for (i = start + 1; i < n ; i++){
        part_sum[i] = part_sum[i-1] + A[i-1];  
      }
      
    }
  // for (i = 1; i < n; i++) {

  //   printf("part_sum[%d] = %d\n", i,part_sum[i]);
  // }
  prefix_sum[0] = 0;
  for (k = 0; k < num_k; k ++){
    end = (k + 1) * piece;
    prefix_sum[end] = prefix_sum[end-piece] + part_sum[end];
    // printf("end[%d] = %d\n", end,prefix_sum[end]);
  }
  #pragma omp parallel for shared(piece,part_sum,prefix_sum) private(k,i,ind) 
    for (k = 0; k < num_k; k ++){
      for (i = 1; i < piece; i++) {
        ind = k * piece + i ;
        prefix_sum[ind] = prefix_sum[k * piece] + part_sum[ind]; 
      }
    } 
    if (piece * num_k < n-1 ) {
      for (i = piece * num_k + 1; i < n ; i++){
        prefix_sum[i] = prefix_sum[piece * num_k] + part_sum[i];  
      }
      
    }
  // for (long i = 1; i < n; i++) {
  //   printf("prefix_sum[%d] = %d\n", i,prefix_sum[i]);
  // }  
  }
  


int main() {
  int test = 0;

  long N = 100000000;
  if(test == 1){
    N = 32;
  }
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  long* part_sum = (long*) malloc(N * sizeof(long));

  for (long i = 0; i < N; i++) A[i] = rand();
  if(test == 1){
      for (long i = 0; i < N; i++) A[i] = i;
    }
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N,part_sum);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
