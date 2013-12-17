#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "mpi.h"

#define DEBUG
//#define CHECK

// Dumb algorithm, using SUMMA idea
// http://cseweb.ucsd.edu/classes/fa12/cse260-b/Lectures/Lec13.pdf

// We have O(x * y * (z / workers)) time
// And O(y * (z / workers)) memory in each worker

int * CreateVector(int n) {
  int * result = (int *) (malloc(sizeof(int) * n));
  assert(result != 0);
  return result;
}

int ** CreateMatrix(int n, int m) {
  int ** result = (int **) (malloc(sizeof(int *) * n));
  assert(result != NULL);
  for(int i = 0; i < n; i++) {
    result[i] = CreateVector(m);
  }
  return result;
}

int GetRowsCount(int z, int workers, int rank) {
  return (z / workers) + ((rank <= (z % workers)) ? 1 : 0);
}

int main(int argc, char * argv[]) {
  // Initialize MPI
  int rank, size;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int workers = size - 1;
  assert(workers > 0);

  // Matrixes
  int x, y, z;
  int ** A = NULL, ** B = NULL, ** C = NULL;

  if(rank == 0) {
    // Init matrix only in master
    x = 3000;
    y = 3000;
    z = 3000;
  }

  MPI_Bcast(&x, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&y, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&z, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if(rank == 0) {
    // Init matrix only in master
    A = CreateMatrix(x, y);
    B = CreateMatrix(y, z);
    C = CreateMatrix(x, z);

    // Generate matrixes
    for(int i = 0; i < x; i++) {
      for(int j = 0; j < y; j++) {
        A[i][j] = (i + 1) * (j + 1);
      }
    }

    for(int i = 0; i < y; i++) {
      for(int j = 0; j < z; j++) {
        B[i][j] = (i + 1) + (j + 1);
      }
    }

    // Transpose B row-by-row
    int Brow[y];

    // Send B rows to assigned worker
    for(int i = 0; i < z; i++) {
      printf("%d\n", i);
      fflush(stdout);
      for(int j = 0; j < y; j++) {
        Brow[j] = B[j][i];
      }
      // Send B row to worker
      const int target_rank = 1 + (i % workers);
      MPI_Send(Brow, y, MPI_INT, target_rank, 0, MPI_COMM_WORLD);
    }
  }

  // Brows for worker
  // O((z / workers) * y) memory
  int ** Brows = NULL;
  int b_rows_cnt = GetRowsCount(z, workers, rank);

  if(rank != 0) {
    // Receive B rows by workers
    Brows = CreateMatrix(b_rows_cnt, y);
    for(int i = 0; i < b_rows_cnt; i++) {
      MPI_Recv(Brows[i], y, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
  }

  // Aline to broadcast
  // O(y) memory
  int * Aline = CreateVector(y);

  // Cbuf to exchange results between master and workers
  // O(z / workers) memory
  const int c_buf_cnt = (z + workers - 1) / workers;
  int * Cbuf = CreateVector(c_buf_cnt);

  for(int i = 0; i < x; i++) {
    // Broadcast A line
    if(rank == 0) {
      #ifdef DEBUG
        fprintf(stderr, "%d\n", i);
      #endif
      memcpy(Aline, A[i], y * sizeof(int));
    }
    MPI_Bcast(Aline, y, MPI_INT, 0, MPI_COMM_WORLD);

    if(rank != 0) {
      for(int i = 0; i < b_rows_cnt; i++) {
        Cbuf[i] = 0;
        for(int j = 0; j < y; j++) {
          Cbuf[i] += Aline[j] * Brows[i][j];
        }
      }
      // Send result to master
      MPI_Send(Cbuf, c_buf_cnt, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
      // Collect results from workers
      for(int j = 0; j < workers; j++) {
        MPI_Recv(Cbuf, c_buf_cnt, MPI_INT, j + 1, 0, MPI_COMM_WORLD, &status);
        // Set up results on right place
        for(int k = j, cursor = 0; k < z; k += workers, ++cursor) {
          C[i][k] = Cbuf[cursor];
        }
      }
    }
  }

  // Check result in master
  #ifdef CHECK
    if(rank == 0) {
      for(int i = 0; i < x; i++) {
        for(int k = 0; k < z; k++) {
          int needed_value = 0;
          for(int j = 0; j < y; j++) {
            needed_value += A[i][j] * B[j][k];
          }
          assert(needed_value == C[i][k]);
        }
      }
    }
  #endif
  
  // Exit
  MPI_Finalize();
  return EXIT_SUCCESS;
}
