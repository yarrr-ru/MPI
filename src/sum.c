#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mpi.h"

int main(int argc, char * argv[]) {
  // Lab settings
  const int arr_size = 3579;
  int arr[arr_size];
  for(int i = 0; i < arr_size; i++) {
    arr[i] = i + 1;
  }

  // Initialize MPI
  int rank, size;
  const int root_rank = 0;
 
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Calculate
  MPI_Status status;
  int result = 0;

  if(arr_size == 1) {
    if(rank == root_rank) {
      result = arr[0];
    }
  } else {
    // Send first messages
    for(int i = rank; i < arr_size; i += size) {
      const int target_rank = (i / 2) % size;
      MPI_Send(&arr[i], 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
    }

    // Go-go tree
    for(int it = 2; it < 2 * arr_size; it *= 2) {
      for(int i = it * rank; i < arr_size; i += it * size) {
        const int left_rank = (i / (it / 2)) % size;

        int left_sum = 0, right_sum = 0;
        MPI_Recv(&left_sum, 1, MPI_INT, left_rank, 0, MPI_COMM_WORLD, &status);

        if((i + (it / 2)) < arr_size) {
          const int right_rank = (left_rank + 1) % size;
          MPI_Recv(&right_sum, 1, MPI_INT, right_rank, 0, MPI_COMM_WORLD, &status);
        }

        result = left_sum + right_sum;
        const int target_rank = (i / (2 * it)) % size;
        MPI_Send(&result, 1, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
      }
    }
  }

  // Output result from root thread
  if(rank == root_rank) {
    printf("%d\n", result);
  }

  // Exit
  MPI_Finalize();
  return EXIT_SUCCESS;
}
