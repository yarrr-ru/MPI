#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mpi.h"

int log_2(int n) {
  int result = 1;
  while(result * 2 < n) {
    result *= 2;
  }
  return result;
}

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
      MPI_Send(&arr[i], 1, MPI_INT, target_rank, i, MPI_COMM_WORLD);
    }

    // Go-go tree
    for(int it = 2; it < arr_size; it *= 2) {
      for(int i = it * rank; i < arr_size; i += it * size) {
        const int left_tag = i;
        const int left_rank = (left_tag / (it / 2)) % size;
        const int right_tag = i + (it / 2);
        const int right_rank = (right_tag / (it / 2)) % size;

        int left_sum = 0, right_sum = 0;
        MPI_Recv(&left_sum, 1, MPI_INT, left_rank, left_tag, MPI_COMM_WORLD, &status);

        if(right_tag < arr_size) {
          MPI_Recv(&right_sum, 1, MPI_INT, right_rank, right_tag, MPI_COMM_WORLD, &status);
        }

        int result_sum = left_sum + right_sum;
        const int target_rank = (i / (2 * it)) % size;
        MPI_Send(&result_sum, 1, MPI_INT, target_rank, i, MPI_COMM_WORLD);
      }
    }

    // Last step
    if(rank == root_rank) {
      const int left_tag = 0;
      const int left_rank = 0;
      const int right_tag = log_2(arr_size);
      const int right_rank = 1;

      int left_sum = 0, right_sum = 0;
      MPI_Recv(&left_sum, 1, MPI_INT, left_rank, left_tag, MPI_COMM_WORLD, &status);
      MPI_Recv(&right_sum, 1, MPI_INT, right_rank, right_tag, MPI_COMM_WORLD, &status);

      result = left_sum + right_sum;
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