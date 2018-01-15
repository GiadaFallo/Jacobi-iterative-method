#include <iostream>

#include "../include/jacobi_data.hpp"
#include "../include/jacobi_seq_solver.hpp"
#include "../include/jacobi_solver.hpp"
#include "../include/jacobi_thread_solver.hpp"
#include "../include/jacobi_ff_solver.hpp"

#define RAND_RANGE_MIN -10.0
#define RAND_RANGE_MAX 10.0
#define SEED 123

enum COMPUTATION_TYPE { SEQUENTIAL, FASTFLOW, THREADS };

void init_matrix(float **A, int matrix_order) {

  for (int i = 0; i < matrix_order; i++) {
    A[i] = new float[matrix_order];
    if (A[i] == NULL) {
      std::cerr << "Error while allocating resources." << std::endl;
      exit(-1);
    }
  }
}

/**
    Generates a random number in a specific range.

    @param fMin The lower bound of the range.
    @param fMax The upper bound of the range.
    @return The generated number.
*/
float generate_random_number(float fMin, float fMax) {
  float f = (float)rand() / RAND_MAX;

  return fMin + f * (fMax - fMin);
}

/**
    Generates a random square matrix.
    @param A Matrix.
    @param matrix_order  Size of matrix.
*/
void generate_random_matrix(float **A, int matrix_order) {

  for (int i = 0; i < matrix_order; i++) {
    float sum = 0.0;
    for (int j = 0; j < matrix_order; j++)
      if (j != i) {
        float val = generate_random_number(RAND_RANGE_MIN, RAND_RANGE_MAX);
        sum += abs(val);

        A[i][j] = val;
      }

    /* Change back A[i][i] to be > then sum(A[i][j]) */
    A[i][i] = sum + generate_random_number(1.0, RAND_RANGE_MAX);
  }
}

/**
    Generates a random vector.
    @param v Vector.
    @param dim  Size of vector.    
*/
void generate_random_vector(float *v, int dim) {
  /* generate vector v */
  for (int j = 0; j < dim; j++) {
    float val = generate_random_number(RAND_RANGE_MIN, RAND_RANGE_MAX);
    v[j] = val;
  }
}

/** Generate a random number in a specific range.
    @param x vector resulting from matrix vector multiplication.
    @param A The square matrix.
    @param v The vector.
    @param matrix_order Size of matrix.
*/
void matrix_vector_multiplication(float *x, float **A, float *v, int matrix_order) {

  for (int i = 0; i < matrix_order; i++) {
    x[i] = 0;
    for (int j = 0; j < matrix_order; j++)
      x[i] += A[i][j] * v[j];
  }
  return;
}


int main(int argc, char *argv[]) {

  // Parse input arguments
  if (argc == 1 || argc == 2 || argc == 3) {
    std::cerr << "Usage is: " << argv[0]
              << "type of iteration (s, t or f), matrix order, maximum number of iterations and epsilon"
              << std::endl;
    return (-1);
  }

  COMPUTATION_TYPE method = SEQUENTIAL;

  switch (argv[1][0]) {
    case ('s'): {
      method = SEQUENTIAL;
    } break;
    case ('f'): {
      method = FASTFLOW;
    } break;
    case ('t'): {
      method = THREADS;
    } break;
  }

  int threads_number = 1;
  int grain = 1;

  switch (method) {
    case (THREADS): {
      if (argc < 5){
        std::cerr << "Usage is: " << argv[0]
        << " type of iteration (s, t or f), matrix order, maximum number of iterations, epsilon and number of threads"
        << std::endl;
        return(-1);
    }
      else
        threads_number = atoi(argv[5]);
    } break;
    case (FASTFLOW): {
      if (argc < 6){
        std::cerr << "Usage is: " << argv[0]
        << " type of iteration (s, t or f), matrix order, maximum number of iterations, epsilon, number of threads and size of grain"
        << std::endl;
        return(-1);
      }
      else {
        threads_number = atoi(argv[5]);
        grain = atoi(argv[6]);
      }
    }
  }

  const int matrix_order = atoi(argv[2]); // order of the matrix
  const int max_iter = atoi(argv[3]);     // number of iterations
  const float epsilon = atof(argv[4]);    // precision

  float *rand_x = new float[matrix_order];
  float *b = new float[matrix_order];
  float **A = new float *[matrix_order];
  float **x = new float *;  

  if (A == NULL || b == NULL || rand_x == NULL) {
    std::cerr << "Error while allocating resources." << std::endl;
    exit(-1);
  }

  init_matrix(A, matrix_order);
  srand(SEED);
  // Generates random matrix and random vector
  generate_random_matrix(A, matrix_order);
  generate_random_vector(rand_x, matrix_order);
  // Computes b vector of system Ax = b
  matrix_vector_multiplication(b, A, rand_x, matrix_order);

  JacobiSolver *js = NULL;

  if (method == SEQUENTIAL)
    js = new JacobiSeqSolver((const float **)A, (const float *)b, matrix_order, threads_number);
  else if (method == THREADS)
    js = new JacobiThreadSolver((const float **)A, (const float *)b, matrix_order, threads_number);
  else if (method == FASTFLOW) 
    js = new JacobiFastFlowSolver((const float **)A, (const float *)b, matrix_order, threads_number, grain);


  // Solve the problem
  (*x) = new float[matrix_order];
  for (int i = 0; i < matrix_order; i++) (*x)[i] = 0.0;

  // Collects and show data
  JacobiData data = js->solve(max_iter, epsilon, x);
  std::cout << data << std::endl;

  // Release resources
  for (int i = 0; i < matrix_order; i++)
    delete[] A[i];

  delete[] A;
  delete[] b;
  delete[] rand_x;
  delete[] x;
  delete js;

  return 0;
}