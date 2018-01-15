#ifndef JACOBI_SOLVER_H
#define JACOBI_SOLVER_H

#include <chrono>
#include <cmath>
#include <iostream>

#include "jacobi_data.hpp"

class JacobiSolver {

  void init_vector(float *v, float x, int dim) {
    for (int i = 0; i < dim; i++) {
        v[i] = x;
    }
    return;
  }
  
protected:
  const float **mA; // Matrix A
  const float *vb; // Vector b
  float **support;
  int matrix_order;
  int max_iter;
  float epsilon;
  int threads_number;
  int repetitions = 20; // number of repetitions 

  JacobiData jdata;
  // The following method will be implemented in all the sub-classes of the template method pattern
  virtual void computation(float **support, float **solution, bool conv, int *k) = 0;
  // The following method help the Jacobi computation
  virtual void update(float **dest, float **source); // update the new values of the solution vector
  virtual bool check_convergence(int k, int max_iter, float epsilon, float *a, float *b); // check if the convergence has been reached
  virtual float compute_precision(float *a, float *b); // compute precision during the computation
  virtual float error_on_computation(float *x); // compute the error
  virtual std::chrono::duration<double> 
    delta_time(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> end); // compute the time needed for the computation

public:
  JacobiSolver(const float **A, const float *b, int n, int t): mA(A), vb(b), matrix_order(n), threads_number(t) { 
    support = new float *;
    (*support) = new float[matrix_order];
    init_vector(*support, 0.0, matrix_order);
  }
  
  ~JacobiSolver() { delete[] support; }
  // method to solve the problem
  JacobiData solve(int iterations, float eps, float **x);
};

#endif