// g++ -fopenmp -O3 -std=c++11 -march=native gs2D-omp.cpp 

#include <stdio.h>
#include <cmath>
#include <iostream>
#include "utils.h"
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;



int N;


double compute_residual_2d(double *u, int N, double invhsq)
{
  int i;
  int j;
  double tmp, res = 0.0;
  #pragma omp parallel for default(none) shared(u,N,invhsq) private(i,j,tmp) reduction(+:res)

  for (i = 1; i <= N; i++){
    for (j = 1; j <= N; j++){
        tmp = ((2.0* u[(N + 2) * i + j]-u[(N + 2) * i + j-1]-u[(N + 2) * i + j+1])*invhsq -1);
    res += tmp * tmp;
  }
}
  res = sqrt(res);
  return res;
}


int main(int argc, const char * argv[]) {

    
    
    # pragma omp parallel
    {
    #ifdef _OPENMP
        int my_threadnum = omp_get_thread_num();
        int numthreads = omp_get_num_threads();
        
    #else
        int my_threadnum = 0;
        int numthreads = 1;
    #endif
    }
    
    cout << "Input N \n";
    cin >> N;
    
   
    
    Timer t;
    t.tic();
    
    double *u = new double[(N+2)*(N+2)]();
    double *unew = new double[(N+2)*(N+2)]();
    
    
    double res, res_init, tol = 1e-5;
    double h = 1.0 / (N + 1);
    int max_iters = 200;
    double hsq = h * h;
    double invhsq = 1./hsq;
    
    
    res_init = compute_residual_2d(u, N, invhsq);
    res = res_init;
    int i;
    int j;
    
    for (int iter = 0; iter < max_iters && res/res_init > tol; iter++) {
      for ( i = 1; i <= N;  i++){
            if (i % 2 == 0){
                j = 2;
            }else {
                j = 1;
            }
            #pragma omp parallel for default(none) shared(N,unew,u,hsq, j,i)
            for (int m = j; m <= N; m = m + 2){
                unew[(N + 2) * i + m] = 0.25 *(hsq  + u[(N + 2) * (i - 1) + m] + u[(N + 2) * i + m - 1] + u[(N + 2) * (i + 1) + m] + u[(N + 2) * i + m + 1] );
            }
            
        }
        

        for ( i = 1; i <= N;  i++){
            if (i % 2 == 0){
                j = 1;
            }else {
                j = 2;
            }
            #pragma omp parallel for default(none) shared(N,unew,u,hsq,j,i)
            for (int m = j; m <= N; m = m + 2){
               unew[(N + 2) * i + m] = 0.25 *(hsq  + u[(N + 2) * (i-1) + m] + u[(N + 2) * i + m - 1] + u[(N + 2) * (i+1) + m] + u[(N + 2) * i + m + 1] );
            }
            
        }
        #pragma omp parallel for default(none) shared(N,unew,u) collapse(2)
        for ( i= 1 ; i <= N; i++){
            for ( j = 1; j <= N; j++){
                u[(N + 2) * i + j] = unew[(N + 2) * i + j];
            }
        }
        
        
        
        
        if ((iter % 20 == 0)) {
            res = compute_residual_2d(u, N, invhsq);
            cout << "Iter = "<<iter<< ", Residual = "<< res<<endl;
        }
    }
         


    double elapsed = t.toc();
    printf("Time elapsed is %fs.\n", elapsed);
    return 0;



}