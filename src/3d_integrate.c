#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "mpi.h"

double f(const double x, const double y, const double z) {
  return (2 * x * y * z + 3 * x * y + 5 * x * z + 7 * y * z
          + 11 * x + 13 * y + 15 * z + 17);
}

int main(int argc, char * argv[]) {
  // Lab settings
  const double x_start = -3, x_end = 7;
  const double y_start = 11, y_end = 17;
  const double z_start = -10, z_end = -5;
  assert(x_start <= x_end && y_start <= y_end && z_start <= z_end);

  const int x_segments = (1 << 8), y_segments = (1 << 7), z_segments = (1 << 6);
  assert(x_segments > 0 && y_segments > 0 && z_segments > 0);

  const double dx = (x_end - x_start) / x_segments;
  const double dy = (y_end - y_start) / y_segments;
  const double dz = (z_end - z_start) / z_segments;
 
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
      for(int k = 0; k < z_segments; k++) {
        const double z_cur = z_start + dz * k;
        result_cur += f(x_cur + dx * 0.5, y_cur + dy * 0.5, z_cur + dz * 0.5) * dx * dy * dz;
      }
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
