#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "mpi.h"

int main( int argc, char * argv[] ) {
  // Initialize MPI
  int rank, size;
  const int root_rank = 0;
 
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Exit
  MPI_Finalize();
  return EXIT_SUCCESS;
}