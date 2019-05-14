// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);


  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the MPI_COMM_WORLDand line
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  int* splitters = (int*)malloc((p-1)*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  std::sort(vec, vec+N);
  for (int i = 0; i < p - 1; i ++){
    splitters[i] = vec[(i+1) * N / p];
  }
  int *all_splitters = NULL;

  if (rank == 0) {
    all_splitters = (int*)malloc(sizeof(int) * p * (p-1));
  }
  MPI_Gather(splitters, p-1, MPI_INT, all_splitters, p-1, MPI_INT, 0,
           MPI_COMM_WORLD);

  int* sp_sorted = (int*)malloc(sizeof(int) * (p-1));

  double tt = MPI_Wtime();
  if (rank == 0) {
    // for (int i=0;i<p*(p-1);i++){printf("all_splitters = %d \n", all_splitters[i]);}
    std::sort(all_splitters, all_splitters+p*(p-1));
    for (int i=0; i<p-1;i++){sp_sorted[i] = all_splitters[(i + 1) * (p - 1)];}
  } 
  MPI_Bcast(sp_sorted, p-1, MPI_INT, 0, MPI_COMM_WORLD);


  int* split_inds = (int*)malloc(sizeof(int) * p);
  split_inds[0] = 0;
  for (int i=0; i<p-1; i++) {
    split_inds[i+1] = std::lower_bound(vec, vec+N, sp_sorted[i]) - vec;
  }

  int*num  = (int*)malloc(sizeof(int) * p);
  int*num_rec = (int*)malloc(sizeof(int) * p);


  for (int i = 0; i < p-1; i++){
    num[i] = split_inds[i+1] - split_inds[i];
  }

  num[p-1] = N - split_inds[p-1];

  MPI_Alltoall(num, 1, MPI_INT, num_rec, 1, MPI_INT, MPI_COMM_WORLD);
  int* rdispls = (int*)malloc(sizeof(int) * p);
  rdispls[0] = 0;

  for (int i=0; i<p-1; i++) {
    rdispls[i+1] = rdispls[i] + num_rec[i];
  }
  int bucs = rdispls[p-1] + num_rec[p-1];
  int* sorted = (int*)malloc(sizeof(int) * bucs);
  MPI_Alltoallv(vec, num, split_inds, MPI_INT, sorted, num_rec, rdispls, MPI_INT, MPI_COMM_WORLD);

  std::sort(sorted, sorted+bucs);
  MPI_Barrier(MPI_COMM_WORLD);

  double elapsed = MPI_Wtime() - tt;
  
  if (rank == 0) printf("Time elapsed is %f seconds.\n", elapsed);

  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename, "w+");

  if(NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }

  for (int i = 0; i < bucs; i++) {
    fprintf(fd, "%d \n", sorted[i]);
  }
  fclose(fd);
  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector

  // every process MPI_COMM_WORLDunicates the selected entries to the root
  // process; use for instance an MPI_Gather

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)

  // root process broadcasts splitters to all other processes

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // split_inds[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data

  // do a local sort of the received data

  // every process writes its result to a file
  free(vec);
  free(all_splitters);
  free(splitters);
  free(sorted);
  free(num);
  free(rdispls);
  free(num_rec);
  free(sp_sorted);
  MPI_Finalize();
  return 0;
}
