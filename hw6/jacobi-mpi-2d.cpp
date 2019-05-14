/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has N unknowns, each processor works with its
 * part, which has lN = N/p unknowns.
 * Author: Georg Stadler
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
MPI_Status status;

MPI_Request request_out1, request_in1;
MPI_Request request_out2, request_in2;
MPI_Request request_out3, request_in3;
MPI_Request request_out4, request_in4;

/* compuate global residual, assuming ghost values are updated */
double compute_residual(double *lu, int lN, double invhsq){
  int i;
  double tmp, gres = 0.0, lres = 0.0;

  for (int i = 1; i <= lN; i++){
    for (int j = 1; j <= lN; j++){
        tmp = ((2.0* lu[(lN + 2) * i + j]-lu[(lN + 2) * i + j-1]-lu[(lN + 2) * i + j+1])*invhsq -1);
    lres += tmp * tmp;
    }
  }
  /* use allreduce for convenience; a reduce would also be sufficient */
  MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  return sqrt(gres);
}



int main(int argc, char * argv[]){
  int rank, i, p, N, lN, iter, max_iters;
  MPI_Status status, status1;
  MPI_Request request_out1, request_in1;
  MPI_Request request_out2, request_in2;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);



  /* get name of host running MPI process */
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  printf("Rank %d/%d running on %s.\n", rank, p, processor_name);

  sscanf(argv[1], "%d", &N);
  sscanf(argv[2], "%d", &max_iters);

  /* compute number of unknowns handled by each process */
  lN = N / (int)sqrt(p);
  if ((N % p != 0) && rank == 0 ) {
    printf("N: %d, local N: %d\n", N, lN);
    printf("Exiting. N must be a multiple of p\n");
    MPI_Abort(MPI_COMM_WORLD, 0);
  }
  
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double tt = MPI_Wtime();

  /* Allocation of vectors, including left/upper and right/lower ghost points */
  double * lu    = (double *) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double * lunew = (double *) calloc(sizeof(double), (lN + 2) * (lN + 2));
  double * lutemp;

  double h = 1.0 / (N + 1);
  double hsq = h * h;
  double invhsq = 1./hsq;
  double gres, gres0, tol = 1e-5;

  int rp = (int)sqrt(p); 
  /* initial residual */
  gres0 = compute_residual(lu, lN, invhsq);
  gres = gres0;

  double * top    = (double *) calloc(sizeof(double), lN);
  double * bottom    = (double *) calloc(sizeof(double), lN);
  double * left    = (double *) calloc(sizeof(double), lN);
  double * right    = (double *) calloc(sizeof(double), lN);

  int row_ind = (int)(rank / rp);
  int col_ind = (int)(rank % rp);


  for (int iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {
    /* communicate ghost values */
    row_ind = (int)(rank / rp);
    col_ind = (int)(rank % rp);

    //send and receive
    // not at the top, receive info from the lunew above, bottom here is the part above the lunew
    if (row_ind != 0){
        for (int i=0;i<lN;i++) top[i] = lunew[lN + 2 + 1 + i];
        MPI_Isend(top, lN, MPI_DOUBLE, rank - rp, rank, MPI_COMM_WORLD, &request_out1); 
        MPI_Irecv(bottom, lN, MPI_DOUBLE, rank - rp, rank - rp, MPI_COMM_WORLD, &request_in2);  
    }
    // not at the bottom, receive info from the lunew below, top here is the part below the lunew
    if (row_ind != rp - 1){
        for (int i=0;i<lN;i++) bottom[i] = lunew[(lN + 2) * (rp - 1 + 1) + 1 + i];
        MPI_Isend(bottom, lN, MPI_DOUBLE, rank + rp, rank, MPI_COMM_WORLD, &request_out2); 
        MPI_Irecv(top, lN, MPI_DOUBLE, rank + rp, rank + rp, MPI_COMM_WORLD, &request_in1); 
    }
    // not at the left side, receive info from the lunew left, right here is the part left of the lunew
    if (col_ind != 0){
        for (int i=0;i<lN;i++) left[i] = lunew[(lN + 2) * (i + 1) + 1];
        MPI_Isend(left, lN, MPI_DOUBLE, rank - 1, rank, MPI_COMM_WORLD, &request_out3);
        MPI_Irecv(right, lN, MPI_DOUBLE, rank - 1, rank - 1, MPI_COMM_WORLD, &request_in4);  
    }
    // not at the right side, receive info from the lunew right, left here is the part right of the lunew
    if (col_ind != rp - 1){
        for (int i=0;i<lN;i++) right[i] = lunew[(lN + 2) * (i + 1) + lN];
        MPI_Isend(right, lN, MPI_DOUBLE, rank + 1, rank, MPI_COMM_WORLD, &request_out4); 
        MPI_Irecv(left, lN, MPI_DOUBLE, rank + 1, rank + 1, MPI_COMM_WORLD, &request_in3);  
    }

    if (row_ind != 0){
        MPI_Wait(&request_out1, &status);
        MPI_Wait(&request_in2, &status);
    }
    if (row_ind != rp - 1){
        MPI_Wait(&request_out2, &status);
        MPI_Wait(&request_in1, &status);
    }
    if (col_ind != rp - 1){
        MPI_Wait(&request_out4, &status);
        MPI_Wait(&request_in3, &status);
    }
    if (col_ind != 0){
        MPI_Wait(&request_out3, &status);
        MPI_Wait(&request_in4, &status);
    }
    if (row_ind != rp - 1){
        for (int i = 0; i < lN; i++) {lunew[(lN + 2) * (rp + 1) + 1 + i] = top[i];}
    }
    if (row_ind != 0){
        for (int i = 0; i < lN; i++) {lunew[1 + i] = bottom[i];}
    }
    if (col_ind != rp - 1){
        for (int i = 0; i < lN; i++) {lunew[(lN + 2) * (1 + i) + lN + 1] = left[i];}
    }
    if (col_ind != 0){
        for (int i = 0; i < lN; i++) {lunew[(lN + 2) * (1 + i)] = right[i];}
    }

    for (int i = 1; i <= lN; i++){
        for (int j = 1; j <= lN; j++){
            lunew[(lN + 2) * i + j] = 0.25 *(hsq  + lu[(lN + 2) * (i - 1) + j] + lu[(lN + 2) * i + j - 1] + lu[(lN + 2) * (i + 1) + j] + lu[(lN + 2) * i + j + 1] );
        }    
    }

  }
  /* Clean up */
  free(lu);
  free(lunew);
  free(top);
  free(bottom);
  free(left);
  free(right);
  /* timing */
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == rank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }
  MPI_Finalize();
  return 0;
}
