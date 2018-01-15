#include "../include/jacobi_solver.hpp"

/**
    Solves system Ax = b with Jacobi iteration method.
    @param iterations The maximum number of iterations to be performed.
    @param eps  The precision to be reached.
    @param solution The solution vector

     @return JacobiData.
*/
JacobiData JacobiSolver::solve(int iterations, float eps, float **solution){  
    
    max_iter = iterations; 
    epsilon = eps;

    std::chrono::time_point<std::chrono::system_clock> start_time, end_time, start_conv, end_conv;  

    int k;
    bool convergence;

    for (int r = 0; r < repetitions; r++) {
        convergence = false; 
        k = 0;
        init_vector(*solution, 0.0, matrix_order);
        init_vector(*support, 0.0, matrix_order);        
        
        start_time = std::chrono::system_clock::now();

        computation(support, solution, convergence, &k);
        
        end_time = std::chrono::system_clock::now();

        jdata.completion_times[r] = (delta_time(start_time, end_time)).count();
    }
    
    jdata.precision = compute_precision(*solution, *support);  
    jdata.iteration_numbers = k;
    jdata.threads = threads_number;
    jdata.error_on_computation = error_on_computation(*solution);

    return jdata;
}


/**
    Updates vector dest with source.
    @param dest The destination vector to be updated.
    @param source  The source vector used to update the destination one.
*/

void JacobiSolver::update(float **dest, float **source){
    float* tmp = *source;
    *source = *dest;
    *dest = tmp;
}

/**
    Updates vector dest with source.
    @param k The current iteration number.
    @param max_iter  The smaximum number of iterations.
    @param epsilon The precision to be reached.
    @param a, b vectors to be compared.

    @return boolean value, true if the convergence is reached (the precision is less or equal the one we want to get or the maximum number of iterations is reached)
*/
bool JacobiSolver::check_convergence(int k, int max_iter, float epsilon, float* a, float* b){
    return (compute_precision(a, b) <= epsilon || k >= max_iter);
}

/**
    Computes the precision between to vectors (square root of the difference of vectors).
    @param a, b vectors to be compared.

    @return The computed precision.
*/

float JacobiSolver::compute_precision(float* a, float* b){
    float sum = 0.0;
    for (size_t i = 0; i < matrix_order; i++)
      sum = sum + ((a[i] - b[i])*(a[i] - b[i]));
    return sqrt(sum);
}


/**
    Computes the error doing the difference: Ax - b.
    @param x vector computed with Jacobi iteration method.

    @return The computed error.
*/
float JacobiSolver::error_on_computation(float* x) {
    float error, sum = 0.0;
  
    for (size_t i = 0; i < matrix_order; i++) {
      sum = 0;
      for (size_t j = 0; j < matrix_order; j++) {
        sum += mA[i][j] * x[j]; 
      }
      error += abs(sum - vb[i]);
    }
    return error / matrix_order;
}

std::chrono::duration<double> 
JacobiSolver::delta_time(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> end) {
  return end - start;
}
