#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

double compute_fd_residual_serial(int, double*, double*);
double compute_fd_residual_parallel(int, double*, double*);
double lp_diff_norm(double*, double*, int, double);
void print_solutions(int, double*);
double compute_fd_residual(int, double*, double*);

double lp_diff_norm(double* a, double* b, int N, double p) {
    double h = 1.0 / (N+1);
    double norm = 0.0;
    double diff;
    int index;
    for(index = 0; index < N*N; index++) {
        diff = fabs(a[index] - b[index]);
        if(p == INFINITY) {
            if(diff > norm) {
                norm = diff;
            }
        } else {
            norm += h * h * pow(diff, p);
        }
    }
    if(p != INFINITY) {
        norm = pow(norm, 1.0 / p);
    }
    return norm;
}

void print_solution(int N, double *u) {
    int i,j,index;
    for(i = 0; i < N; i++) {
        for(j = 0; j < N; j++) {
            index = N * i + j;
            printf("%3f  ", u[index]);
        }
        printf("\n");
    }
}

double compute_fd_residual(int N, double* u, double* f) {
    #ifdef _OPENMP
    return compute_fd_residual_parallel(N, u, f);
    #else
    return compute_fd_residual_serial(N, u, f);
    #endif 
}

double compute_fd_residual_parallel(int N, double *u, double *f) {
    double resid = 0.0;
    double diff;
    double h = 1.0 / (N + 1);
    double h_sqr = h * h;
    int i,j, index;
    #pragma omp parallel shared(h_sqr, h) private(i,j,index,diff)\
        reduction(+:resid)
    {
        #pragma omp sections nowait
        {
            #pragma omp section
            {
                // First row:
                diff = f[0] + (-4 * u[0] + u[1] + u[N]) / h_sqr;
                resid += diff * diff;
                for(j = 1; j < N-1; j++) {
                    diff = f[j] + (-4 * u[j] + u[j-1] + u[j+1] + u[j+N])\
                                    / h_sqr;
                    resid += diff * diff;
                }
                diff = f[N-1] + (-4 * u[N-1] + u[N-2] + u[2*N-1]) / h_sqr;
                resid += diff * diff;
            }

            #pragma omp section
            {
                // Last row:
                index = N*(N-1);
                diff = f[index] + (u[index + 1] + u[index - N] - 4 * u[index])\
                                    / h_sqr;
                resid += diff * diff;
                for(j = 1; j < N-1; j++) {
                    index = N*(N-1) + j;
                    diff = f[index] + (u[index - 1] + u[index + 1]\
                                + u[index - N] - 4 * u[index]) / h_sqr;
                    resid += diff * diff;
                }
                index = N * N - 1;
                diff = f[index] + (u[index - 1] + u[index - N] - 4 * u[index])\
                                / h_sqr;
                resid += diff * diff;
            }

        }
        // Interior rows:
        #pragma omp for
        for(i = 1; i < N - 1; i++){
            index = N * i;
            diff = f[index] + (u[index + 1] + u[index + N] + u[index - N] \
                        - 4 * u[index]) / h_sqr;
            resid += diff * diff;
            for(j = 1; j < N - 1; j++) {
                index = N * i + j;
                diff = f[index] + (u[index + 1] + u[index - 1] + u[index + N] \
                            + u[index - N] - 4 * u[index]) / h_sqr;
                resid += diff * diff;
            }
            index = N * i + (N - 1);
            diff = f[index] + (u[index - 1] + u[index - N] + u[index + N] \
                        - 4 * u[index]) / h_sqr;
            resid += diff * diff;
        }
    }
    return sqrt(resid);
}

double compute_fd_residual_serial(int N, double* u, double *f) {
    double resid = 0.0;
    double diff;
    double h = 1.0 / (N + 1);
    double h_sqr = h * h;
    int i,j, index;
    // First row:
    diff = f[0] + (-4 * u[0] + u[1] + u[N]) / h_sqr;
    resid += diff * diff;
    for(j = 1; j < N-1; j++) {
        diff = f[j] + (-4 * u[j] + u[j-1] + u[j+1] + u[j+N]) / h_sqr;
        resid += diff * diff;
    }
    diff = f[N-1] + (-4 * u[N-1] + u[N-2] + u[2*N-1]) / h_sqr;
    resid += diff * diff;

    // Interior rows:
    for(i = 1; i < N - 1; i++){
        index = N * i;
        diff = f[index] + (u[index + 1] + u[index + N] + u[index - N] \
                    - 4 * u[index]) / h_sqr;
        resid += diff * diff;
        for(j = 1; j < N - 1; j++) {
            index = N * i + j;
            diff = f[index] + (u[index + 1] + u[index - 1] + u[index + N] \
                        + u[index - N] - 4 * u[index]) / h_sqr;
            resid += diff * diff;
        }
        index = N * i + (N - 1);
        diff = f[index] + (u[index - 1] + u[index - N] + u[index + N] \
                    - 4 * u[index]) / h_sqr;
        resid += diff * diff;
    }

    // Last row:
    index = N*(N-1);
    diff = f[index] + (u[index + 1] + u[index - N] - 4 * u[index]) / h_sqr;
    resid += diff * diff;
    for(j = 1; j < N-1; j++) {
        index = N*(N-1) + j;
        diff = f[index] + (u[index - 1] + u[index + 1] + u[index - N] \
                    - 4 * u[index]) / h_sqr;
        resid += diff * diff;
    }
    index = N * N - 1;
    diff = f[index] + (u[index - 1] + u[index - N] - 4 * u[index]) / h_sqr;
    resid += diff * diff;
    return sqrt(resid);
}