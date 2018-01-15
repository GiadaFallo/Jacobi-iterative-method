#ifndef JACOBIFFSOLVER_H
#define JACOBIFFSOLVER_H

#include <ff/parallel_for.hpp>
#include "jacobi_solver.hpp"

class JacobiFastFlowSolver : public JacobiSolver{
public:
    JacobiFastFlowSolver(const float** A, const float* b, int matrix_order, int threads_number, int grain);
    virtual ~JacobiFastFlowSolver() {delete pf;}

private:
    int grain;
    ff::ParallelFor* pf;
    void computation(float **x, float **dest, bool convergence, int *k);
    void iteration(float **x, float **dest);
};
#endif