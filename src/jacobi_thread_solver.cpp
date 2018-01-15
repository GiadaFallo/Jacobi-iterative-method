// #include <thread>
// #include "../include/jacobi_thread_solver.hpp"

// void JacobiThreadSolver::computation(float **support, float **solution, bool convergence, int *k) {

//     while (!convergence){


//         std::thread *threads = new std::thread[threads_number];
        
//         if (threads == NULL) {
//         std::cerr << "Error while allocating resources." << std::endl;
//         exit(-1);
//         }
    
//         int quotient = ceil(matrix_order / threads_number);
//         int mod = matrix_order % threads_number;
    
//         size_t start = 0;
//         size_t end = 0;
//         size_t w = 0;
        
//         while (start < matrix_order) {
    
//         end = start + quotient - 1;
//         end = (mod-- > 0) ? end + 1 : end;
    
//         threads[w++] = std::thread(&JacobiThreadSolver::iter, this, start, end, solution, support);
    
//         start = end + 1;
//         }
    
//         for (int w = 0; w < threads_number; w++)
//             threads[w].join();
    
//         delete[] threads;
        
//         update(solution, support);
//         (*k)++;
    
//         // start_conv = std::chrono::system_clock::now();
//         convergence = check_convergence(*k, max_iter, epsilon, *solution, *support);
//         // end_conv = std::chrono::system_clock::now();
//     }
// }

// void JacobiThreadSolver::iter(size_t start, size_t end, float **solution, float **support) {
//   float sum = 0.0;  
  
//   for (int i = start; i <= end; i++) {
//     sum = vb[i];
//     for (int j = 0; j < i; j++) {
//       sum = sum - mA[i][j] * (*support)[j];
//     }
//     for (int j = i + 1; j < matrix_order; j++) {
//       sum = sum - mA[i][j] * (*support)[j];
//     }
//     (*solution)[i] = sum / mA[i][i];
//   }
//   return;
// }