#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_ring(int* msg, int num_procs, long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  // char* msg = (char*) malloc(Nsize);
  // int msg = 0;
  
  int from;
  int dest;

  // for (long i = 0; i < Nsize; i++) msg[i] = 0;

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    MPI_Status status;
    if (rank == 0)
        MPI_Send(msg, Nsize, MPI_INT, 1, repeat, comm);
    for (int proc = 1; proc < num_procs; proc++){
      if (rank == proc){
        from = 100;
        dest = 100;
        if (proc == num_procs - 1){
          from = proc - 1;
          dest = 0;
        }
        else{
          from = proc - 1;
          dest = proc + 1;
        }
        
        MPI_Recv(msg, Nsize, MPI_INT, from, repeat, comm, &status);
        msg[0] += rank;
        MPI_Send(msg, Nsize, MPI_INT, dest, repeat, comm);
      }  
    }
    if (rank == 0)
      MPI_Recv(msg, Nsize, MPI_INT, num_procs - 1, repeat, comm, &status);   
    
  }
  tt = MPI_Wtime() - tt;
  // free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 3) {
    printf("Usage: mpirun ./int_ring <num_procs> <Nrepeat>\n");
    abort();
  }
  int* msg = (int*) malloc(1);
  msg[0] = 0;
  int num_procs = atoi(argv[1]);
  long Nrepeat = atoi(argv[2]);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  if (rank == 0)
    printf("num_procs = %d, Nrepeat = %ld\n", num_procs, Nrepeat);
  
  double tt = time_ring(msg,num_procs, Nrepeat, 1, comm);
  if (!rank) printf("result is : %d\n", msg[0]);
  if (!rank) printf("ring latency: %e ms\n", tt/Nrepeat * 1000);


  long Nsize = 10e5;
  if (!rank) printf("Nsize is : %ld\n", Nsize);
  int* msg2 = (int*) malloc(Nsize * sizeof(int));
  
  tt = time_ring(msg2,num_procs, Nrepeat, Nsize, comm);
  if (!rank) printf("ring bandwidth: %e GB/s\n", (Nsize*Nrepeat)/tt/1e9);

  MPI_Finalize();
  return 0;
}

