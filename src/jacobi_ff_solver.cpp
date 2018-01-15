#include "../include/jacobi_ff_solver.hpp"

JacobiFastFlowSolver::JacobiFastFlowSolver(const float** A, const float* b, int matrix_order, int threads_number, int grain) : JacobiSolver(A, b, matrix_order, threads_number), grain(grain){
    pf = new ff::ParallelFor(threads_number, true, true);
}

void JacobiFastFlowSolver::iteration(float **support, float **solution) {

  pf->parallel_for(0, matrix_order, 1, grain, 
      [&](const long i) {
        float sum;
        sum = vb[i];
        for (size_t j = 0; j < i; j++)
          sum = sum - mA[i][j] * (*support)[j];
        for (size_t j = i + 1; j < matrix_order; j++)
          sum = sum - mA[i][j] * (*support)[j];
        (*solution)[i] = sum / mA[i][i];
      },
      threads_number);
}


void JacobiFastFlowSolver::computation(float **support, float **solution, bool convergence, int *k){
    
      while (!convergence){
        
          iteration(support, solution);
          update(solution, support);
          (*k)++;
        
          convergence = check_convergence(*k, max_iter, epsilon, *solution, *support);
      }
    
    }