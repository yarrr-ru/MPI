#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "mpi.h"

#define CHECK
//#define DUMB
#define DEBUG

// Awesome simple reference
// https://github.com/yarrr-ru/MPI/blob/master/doc/1_1_%D0%9C%D0%B5%D1%82%D0%BE%D0%B4_%D0%BF%D1%80%D0%BE%D0%B3%D0%BE%D0%BD%D0%BA%D0%B8.pdf

// O(max(workers, n / workers)) time
// O(n / workers) memory in each worker

#define MIN(a, b) ((a < b) ? a : b)

double * CreateVector(int n) {
  double * result = (double *) (malloc(sizeof(double) * n));
  assert(result != 0);
  for(int i = 0; i < n; i++) {
    result[i] = 0.0;
  }
  return result;
}

double RandomValue(int maxValue) {
  int int_part = (rand() % maxValue);
  double frac_part = ((double) rand()) / RAND_MAX;
  int signum = (rand() % 2 == 0) ? 1 : -1;
  return (signum * int_part + frac_part);
}

int GetLinesInWorker(int n, int workers, int lines_per_worker, int id) {
  const int l = lines_per_worker * id;
  return (id + 1 == workers) ? (n - l) : lines_per_worker;
}

double DoubleAbs(double value) {
  return (value < 0) ? -value : value;
}

int DoubleEquals(double a, double b) {
  const static double eps = 1e-4;
  return (DoubleAbs(a - b) < eps);
}

int main(int argc, char * argv[]) {
  // Set time as seed
  srand(0);

  // Initialize MPI
  int rank, size;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int workers = size - 1;
  assert(workers > 0);

  // Matrix
  int n;
  double * A = NULL, * B = NULL, * C = NULL, * F = NULL, * X = NULL;
  double * An = NULL, * Bn = NULL, * Cn = NULL, * Fn = NULL, * Xn = NULL;
  double * El = NULL, * Er = NULL;

  if(rank == 0) {
    // Generate matrix in master
    n = 100;
    A = CreateVector(n);
    B = CreateVector(n);
    C = CreateVector(n);
    F = CreateVector(n);
    X = CreateVector(n);

    for(int i = 0; i < n; i++) {
      A[i] = RandomValue(100);
      F[i] = RandomValue(100);
      if(i < n - 1) {
        C[i] = RandomValue(100);
      } else {
        C[i] = 0.0;
      }
      if(i > 0) {
        B[i] = RandomValue(100);
      } else {
        B[i] = 0.0;
      }
    }

    // Corner cases
    assert(n > 0);

    if(n == 1) {
      assert(!DoubleEquals(A[0], 0.0));
      X[0] = F[0] / A[0];
      workers = 0;
    } else if(2 * workers > n) {
      // Too many workers, we need at least 2 lines for each
      workers = (n / 2);
    }
  }

  // Bcast workers count to workers
  MPI_Bcast(&workers, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // We dont need this worker anymore
  if(rank > workers) {
    MPI_Finalize();
    return EXIT_SUCCESS;
  }

  // Need to do something more
  if(workers > 0) {
    // Bcast matrix size to workers
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const int lines_per_worker = n / workers;
    const int lines_in_this_worker =
      (rank == 0) ? 0 : GetLinesInWorker(n, workers, lines_per_worker, rank - 1);

    if(rank == 0) {
      // Send part of matrix to workers
      for(int i = 0; i < workers; i++) {
        const int left_border = i * lines_per_worker;
        const int take = GetLinesInWorker(n, workers, lines_per_worker, i);
        MPI_Send((A + left_border), take, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
        MPI_Send((B + left_border), take, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
        MPI_Send((C + left_border), take, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
        MPI_Send((F + left_border), take, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
      }
    } else {
      // Receive part of matrix in workers
      A = CreateVector(lines_in_this_worker);
      B = CreateVector(lines_in_this_worker);
      C = CreateVector(lines_in_this_worker);
      F = CreateVector(lines_in_this_worker);
      MPI_Recv(A, lines_in_this_worker, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(B, lines_in_this_worker, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(C, lines_in_this_worker, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(F, lines_in_this_worker, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

      // First part, eliminations
      El = CreateVector(lines_in_this_worker);
      Er = CreateVector(lines_in_this_worker);

      // First elimination
      for(int i = 0; i < lines_in_this_worker - 1; i++) {
        assert(!DoubleEquals(A[i], 0.0));
        const double m = B[i + 1] / A[i];
        El[i + 1] -= m * (B[i] + El[i]);
        B[i + 1] = 0.0;
        A[i + 1] -= m * C[i];
        F[i + 1] -= m * F[i];
      }

      // Second elimination
      for(int i = lines_in_this_worker - 1; i > 0; i--) {
        assert(!DoubleEquals(A[i], 0.0));
        const double m = C[i - 1] / A[i];
        El[i - 1] -= m * El[i];
        C[i - 1] = 0.0;
        Er[i - 1] -= (Er[i] + C[i]) * m;
        F[i - 1] -= m * F[i];
      }

      // Send results from top and bottom line to master
      El[0] += B[0];
      Er[lines_in_this_worker - 1] += C[lines_in_this_worker - 1];

      // Top line
      MPI_Send(&El[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&A[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&Er[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&F[0], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

      // Bottom line
      MPI_Send(&El[lines_in_this_worker - 1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&A[lines_in_this_worker - 1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&Er[lines_in_this_worker - 1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
      MPI_Send(&F[lines_in_this_worker - 1], 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Get top/bottom lines from each worker
    if(rank == 0) {
      An = CreateVector(2 * workers);
      Bn = CreateVector(2 * workers);
      Cn = CreateVector(2 * workers);
      Fn = CreateVector(2 * workers);
      Xn = CreateVector(2 * workers);

      // Collect results
      for(int i = 0; i < workers; i++) {
        // Top worker line
        MPI_Recv(&Bn[2 * i], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&An[2 * i], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&Cn[2 * i], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&Fn[2 * i], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);

        // Bottom worker line
        MPI_Recv(&Bn[2 * i + 1], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&An[2 * i + 1], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&Cn[2 * i + 1], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&Fn[2 * i + 1], 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status);
      }

      // Fix results
      for(int i = 1; i < 2 * workers - 2; i += 2) {
        assert(!DoubleEquals(Cn[i], 0.0));
        const double m = Cn[i - 1] / Cn[i];
        Bn[i - 1] -= m * Bn[i];
        Cn[i - 1] = -m * An[i];
        Fn[i - 1] -= m * Fn[i];
      }

      for(int i = 2; i < 2 * workers - 1; i += 2) {
        assert(!DoubleEquals(Bn[i], 0.0));
        const double m = Bn[i + 1] / Bn[i];
        Bn[i + 1] = -m * An[i];
        An[i + 1] -= m * Cn[i];
        Fn[i + 1] -= m * Fn[i];
      }

      // Use Thomas algorithm on this results
      for(int i = 0; i < 2 * workers - 1; i++) {
        assert(!DoubleEquals(An[i], 0.0));
        const double m = (Bn[i + 1] / An[i]);
        Bn[i + 1] = 0.0;
        An[i + 1] -= m * Cn[i];
        Fn[i + 1] -= m * Fn[i];
      }

      for(int i = 2 * workers - 1; i > 0; i--) {
        assert(!DoubleEquals(An[i], 0.0));
        const double m = Cn[i - 1] / An[i];
        Cn[i - 1] = 0.0;
        Fn[i - 1] -= m * Fn[i];
      }

      for(int i = 0; i < 2 * workers; i++) {
        assert(!DoubleEquals(An[i], 0.0));
        Xn[i] = Fn[i] / An[i];
        An[i] = 1.0;
      }

      // Ok, send results to each worker back
      for(int i = 0; i < workers; i++) {
        double first = (i == 0) ? 0 : Xn[(i - 1) * 2 + 1];
        double second = (i + 1 == workers) ? 0 : Xn[(i + 1) * 2];

        MPI_Send(&first, 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
        MPI_Send(&second, 1, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
      }
    }

    // Ok, now we have part of results in each worker, need to get rest
    if(rank != 0) {
      // Get part of results from master
      double first = 0.0, second = 0.0;
      MPI_Recv(&first, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
      MPI_Recv(&second, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

      // Get rest of results
      X = CreateVector(lines_in_this_worker);

      for(int i = 0; i < lines_in_this_worker; i++) {
        assert(!DoubleEquals(A[i], 0.0));
        X[i] = (F[i] - El[i] * first - Er[i] * second) / A[i];
      }
      
      // Send them to masters
      MPI_Send(X, lines_in_this_worker, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    // Collect results from workers
    if(rank == 0) {
      for(int i = 0; i < workers; i++) {
        const int left_border = i * lines_per_worker;
        const int take = GetLinesInWorker(n, workers, lines_per_worker, i);
        MPI_Recv((X + left_border), take, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, &status); 
      }
    }
  }

  // Print results
  if(rank == 0) {
    printf("OK\n");
    printf("[");
    printf(" %lf ", X[0]);
    for(int i = 1; i < n; i++) {
      printf("; %lf", X[i]);
    }
    printf(" ]\n");
  }

  // Dumb way to solve
  #ifdef DUMB
    if(rank == 0) {
      double * Ac = CreateVector(n);
      double * Bc = CreateVector(n);
      double * Cc = CreateVector(n);
      double * Fc = CreateVector(n);

      for(int i = 0; i < n; i++) {
        Ac[i] = A[i];
        Bc[i] = B[i];
        Cc[i] = C[i];
        Fc[i] = F[i];
      }

      for(int i = 0; i < n - 1; i++) {
        assert(!DoubleEquals(Ac[i], 0.0));
        const double m = (Bc[i + 1] / Ac[i]);
        Bc[i + 1] = 0.0;
        Ac[i + 1] -= m * Cc[i];
        Fc[i + 1] -= m * Fc[i];
      }

      for(int i = n - 1; i > 0; i--) {
        assert(!DoubleEquals(Ac[i], 0.0));
        const double m = Cc[i - 1] / Ac[i];
        Cc[i - 1] = 0.0;
        Fc[i - 1] -= m * Fc[i];
      }

      for(int i = 0; i < n; i++) {
        assert(!DoubleEquals(Ac[i], 0.0));
        X[i] = Fc[i] / Ac[i];
      }
    }
  #endif

  // Check results
  #ifdef CHECK
    if(rank == 0) {
    for(int i = 0; i < n; i++) {
      double current = A[i] * X[i];
      if(i + 1 < n) {
        current += C[i] * X[i + 1];
      }
      if(i > 0) {
        current += B[i] * X[i - 1];
      }
      assert(DoubleEquals(current, F[i]));
    }
  }
  #endif

  // Exit
  MPI_Finalize();
  return EXIT_SUCCESS;
}
