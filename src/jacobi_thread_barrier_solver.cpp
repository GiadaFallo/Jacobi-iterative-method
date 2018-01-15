#include "../include/jacobi_thread_solver.hpp"

#include <thread>

void JacobiThreadSolver::iteration(size_t start, size_t end, float **solution, float **support, int &iter, 
                                                                    float &precision, JacobiBarrier &bar) {
  float sum = 0.0;

  for (size_t k = 1; k <= max_iter and precision >= epsilon; k++) {

    for (int i = start; i <= end; i++) {
      sum = vb[i];
      for (int j = 0; j < i; j++) {
        sum = sum - mA[i][j] * (*support)[j];
      }
      for (int j = i + 1; j < matrix_order; j++) {
        sum = sum - mA[i][j] * (*support)[j];
      }
      (*solution)[i] = sum / mA[i][i];
    }

    bar.arrive_and_wait([&] {
      iter++;
      precision = compute_precision(*solution, *support);
      update(solution, support);
    });
  }
}

void JacobiThreadSolver::computation(float **support, float **solution, bool conv, int *k) {

  float precision = 1.0;
  int iter = 0;
  std::thread *threads = new std::thread[threads_number];

  if (threads == NULL) {
    std::cerr << "Error while allocating resources." << std::endl;
    exit(-1);
  }

  int quotient = ceil(matrix_order / threads_number);
  int mod = matrix_order % threads_number;

  JacobiBarrier bar(threads_number);

  size_t start = 0;
  size_t end = 0;
  size_t w = 0;

  while (start < matrix_order) {

    end = start + quotient - 1;
    end = (mod-- > 0) ? end + 1 : end;

    threads[w++] = std::thread(&JacobiThreadSolver::iteration, this, start, end, solution, support, std::ref(iter), std::ref(precision), std::ref(bar));

    start = end + 1;
  }

  for (int w = 0; w < threads_number; w++)
    threads[w].join();

    *k = iter;

  delete[] threads;
}

