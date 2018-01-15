#ifndef JACOBITHREADSOLVER_H
#define JACOBITHREADSOLVER_H

#include "jacobi_barrier.hpp"
#include "jacobi_solver.hpp"

class JacobiThreadSolver : public JacobiSolver {
public:
  JacobiThreadSolver(const float **A, const float *b, int matrix_order, int threads_number): JacobiSolver(A, b, matrix_order, threads_number) {}

private:
  void computation(float **x, float **dest, bool conv, int *k);
  void iteration(size_t start, size_t end, float **solution, float **support, int&iter, float&precision, JacobiBarrier &bar);
};

#endif