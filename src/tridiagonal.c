#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "mpi.h"

int main(int argc, char * argv[]) {
  // Initialize MPI
  int rank, size;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  

  // Exit
  MPI_Finalize();
  return EXIT_SUCCESS;
}