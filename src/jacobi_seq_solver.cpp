#include "../include/jacobi_seq_solver.hpp"

void JacobiSeqSolver::iteration(float **support, float **solution) {
  float sum = 0.0;
  
  for (int i = 0; i < matrix_order; i++) {

    sum = vb[i];

    for (int j = 0; j < i; j++) {
      sum = sum - mA[i][j] * (*support)[j];
    }   
    for (int j = i + 1; j < matrix_order; j++) { 
      sum = sum - mA[i][j] * (*support)[j];
    }
    (*solution)[i] = sum / mA[i][i];
  }
}


void JacobiSeqSolver::computation(float **support, float **solution, bool convergence, int *k){

  while (!convergence){
    
      iteration(support, solution);
      update(solution, support);
      (*k)++;
    
      convergence = check_convergence(*k, max_iter, epsilon, *solution, *support);
  }

}

