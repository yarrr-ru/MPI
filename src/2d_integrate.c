#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mpi.h"

double f(const double x, const double y) {
  return (x * y + 3 * x + 7 * y + 11);
}

int main(int argc, char * argv[]) {
  // Lab settings
  const double x_start = -3, x_end = 7;
  const double y_start = 11, y_end = 17;
  assert(x_start <= x_end && y_start <= y_end);

  const int x_segments = (1 << 9), y_segments = (1 << 8);
  assert(x_segments > 0 && y_segments > 0);

  const double dx = (x_end - x_start) / x_segments;
  const double dy = (y_end - y_start) / y_segments;

  // Initialize MPI
  int rank, size;
  const int root_rank = 0;
 
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Calculate
  double result = 0.0, result_cur = 0.0;

  for(int i = rank; i < x_segments; i += size) {
    const double x_cur = x_start + dx * i;
    for(int j = 0; j < y_segments; j++) {
      const double y_cur = y_start + dy * j;
      result_cur += f(x_cur + dx * 0.5, y_cur + dy * 0.5) * dx * dy;
    }
  }

  // Send result from this thread to main
  MPI_Reduce(&result_cur, &result, 1, MPI_DOUBLE, MPI_SUM, root_rank, MPI_COMM_WORLD);

  // Output result from root thread
  if(rank == root_rank) {
    printf("%lf\n", result);
  }

  // Exit
  MPI_Finalize();
  return EXIT_SUCCESS;
}