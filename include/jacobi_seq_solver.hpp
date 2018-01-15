#ifndef JACOBISEQSOLVER_H
#define JACOBISEQSOLVER_H

#include "jacobi_solver.hpp"

class JacobiSeqSolver: public JacobiSolver{
public:
    JacobiSeqSolver(const float** A, const float* b, int matrix_order, int threads_number): JacobiSolver(A, b, matrix_order, threads_number) {}
    
private:
    void computation(float **x, float **dest, bool conv, int* k);
    void iteration(float **x, float **dest);

};

#endif