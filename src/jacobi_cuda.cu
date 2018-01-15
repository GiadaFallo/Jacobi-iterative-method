#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<getopt.h>
#include <assert.h>
#include <cuda.h>
#include <chrono>

#define RAND_RANGE_MIN -10.0
#define RAND_RANGE_MAX 10.0
#define SEED 123
#define JACOBI_DEBUG 0

enum ERROR_TYPE { MEMCPY, CMALLOC, ADDK, DEVSYNC };

void init_const(float* v, float x, int dim) {
  for (int i = 0; i < dim; i++)
    v[i] = x;
  return;
}

void init_matrix(float **A, int matrix_order) {

  for (int i = 0; i < matrix_order; i++) {
    A[i] = new float[matrix_order];
    if (A[i] == NULL) {
      std::cerr << "Error while allocating resources." << std::endl;
      exit(-1);
    }
  }
}

/** Generates a random number in a specific range.

    @param fMin The lower bound of the range.
    @param fMax The upper bound of the range.
    @return The generated number.
*/
float generate_random_number(float fMin, float fMax) {
  float f = (float)rand() / RAND_MAX;

  return fMin + f * (fMax - fMin);
}

/** Generates a random square matrix.
    @param A The matrix.
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

/** Generates a random vector.
    @param v .
*/
void generate_random_vector(float *v, int matrix_order) {
  /* generate vector v */
  for (int j = 0; j < matrix_order; j++) {
    float val = generate_random_number(RAND_RANGE_MIN, RAND_RANGE_MAX);
    v[j] = val;
  }
}

/** Generate a random number in a specific range.
    @param A The square matrix.
    @param v The vector.
    @param start .
    @param end .
*/
void matrix_vector_multiplication(float *x, float **A, float *v, int matrix_order) {

  for (int i = 0; i < matrix_order; i++) {
    x[i] = 0;
    for (int j = 0; j < matrix_order; j++)
      x[i] += A[i][j] * v[j];
  }
  return;
}

void error_on_computation(float* x, float ** A, float *b, int matrix_order, float *err) {
    float error = 0.0, sum = 0.0;
  
    for (size_t i = 0; i < matrix_order; i++) {
      sum = 0.0;
      for (size_t j = 0; j < matrix_order; j++) {
          
        sum = sum + A[i][j] * x[j]; 
      }
      error = error + abs(sum - b[i]);
    }
    *err = error / matrix_order;
    return;
}


std::chrono::duration<double> delta_time(std::chrono::time_point<std::chrono::system_clock> start, std::chrono::time_point<std::chrono::system_clock> end) {
  return end - start;
}


cudaError_t error_check(cudaError_t cudaStatus, ERROR_TYPE msgtype, float*dev_a, float*dev_x_solution, float*dev_b, float*dev_prec_values){
	
    if (cudaStatus != cudaSuccess) {

        switch(msgtype) {
            case (CMALLOC):{
                std::cerr <<  "cudaMalloc failed!" << std::endl;
            }
            case (MEMCPY):{
                std::cerr <<  "cudaMemcpy failed!" << std::endl;
            }
            case (ADDK):{
                std::cerr << "addKernel launch failed:" << cudaGetErrorString(cudaStatus) << std::endl;
            }
            case(DEVSYNC):{
                std::cerr << "cudaDeviceSynchronize returned error code " << cudaStatus <<  " after launching jacobi!"  << std::endl;
            }
        }
		
    cudaFree(dev_a);
    cudaFree(dev_x_solution);
    cudaFree(dev_prec_values);
    cudaFree(dev_b);
	
    return cudaStatus;
	}
}

__global__ void iteration(float * a, float * x_solution, float * b, float * prec_values, unsigned int matrix_order) { 
    unsigned int j, i;
	float sigma = 0.0, newValue = 0.0;
    int bx = blockIdx.x, tx = threadIdx.x;
    i = tx + bx*blockDim.x;
    
    if (i >= matrix_order) return;

    if (i < matrix_order){
        sigma = b[i];
        int idx_Ai = i*matrix_order;
        
        for (j = 0; j < matrix_order; j++) {
            if (i != j) { sigma = sigma - a[idx_Ai + j] * x_solution[j]; }
        }

        newValue = sigma / a[idx_Ai + i];
        
        prec_values[i] = (x_solution[i] - newValue)*(x_solution[i] - newValue);
        x_solution[i] = newValue;
        __syncthreads();
    }
}

cudaError_t cuda_jacobi_solve(float * a, float * x_solution, float * b, float eps, unsigned int matrix_order, int * max_iter, float *prec) {
	unsigned int i, j;
	
    int k = 0, nTiles;
	float *dev_a = 0, *dev_x_solution = 0, *dev_b = 0, *dev_prec_values = 0;
    float accur = 1.0, sum = 0.0;

    float *prec_values = new float[matrix_order];
    init_const(prec_values, 0.0, matrix_order);

    size_t matrix_size = matrix_order*matrix_order*sizeof(float);
    size_t vector_size = matrix_order*sizeof(float);

	cudaError_t cudaStatus;

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

	cudaStatus = cudaMalloc((void**)&dev_a, matrix_size);
    error_check(cudaStatus, CMALLOC, dev_a, dev_x_solution, dev_b, dev_prec_values);

	cudaStatus = cudaMalloc((void**)&dev_x_solution, vector_size);
    error_check(cudaStatus, CMALLOC, dev_a, dev_x_solution, dev_b, dev_prec_values);

	cudaStatus = cudaMalloc((void**)&dev_b, vector_size);
    error_check(cudaStatus, CMALLOC, dev_a, dev_x_solution, dev_b, dev_prec_values);

    cudaStatus = cudaMalloc((void**)&dev_prec_values, vector_size);
    error_check(cudaStatus, CMALLOC, dev_a, dev_x_solution, dev_b, dev_prec_values); 

	cudaStatus = cudaMemcpy(dev_a, a, matrix_size, cudaMemcpyHostToDevice);
    error_check(cudaStatus, MEMCPY, dev_a, dev_x_solution, dev_b, dev_prec_values);

	cudaStatus = cudaMemcpy(dev_x_solution, x_solution, vector_size, cudaMemcpyHostToDevice);
    error_check(cudaStatus, MEMCPY, dev_a, dev_x_solution, dev_b, dev_prec_values);

	cudaStatus = cudaMemcpy(dev_b, b, vector_size, cudaMemcpyHostToDevice);
    error_check(cudaStatus, MEMCPY, dev_a, dev_x_solution, dev_b, dev_prec_values);

	cudaStatus = cudaMemcpy(dev_prec_values, prec_values, vector_size, cudaMemcpyHostToDevice);    
    error_check(cudaStatus, MEMCPY, dev_a, dev_x_solution, dev_b, dev_prec_values);    

    int tileSize = 16;
    nTiles = matrix_order/tileSize + (matrix_order%tileSize == 0?0:1);

    for (i = 0; i < *max_iter; i++) {

        iteration <<<nTiles,tileSize>>> (dev_a, dev_x_solution, dev_b, dev_prec_values, matrix_order);
        k++;

        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        error_check(cudaStatus, ADDK, dev_a, dev_x_solution, dev_b, dev_prec_values);

        // cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        error_check(cudaStatus, DEVSYNC, dev_a, dev_x_solution, dev_b, dev_prec_values);

        // Retreive the dev_prec_values vector with all the precision values
        cudaStatus = cudaMemcpy(prec_values, dev_prec_values, vector_size, cudaMemcpyDeviceToHost);
        error_check(cudaStatus, MEMCPY, dev_a, dev_x_solution, dev_b, dev_prec_values);    

        // Computes the precision 
        sum = 0.0;
        for (j = 0; j < matrix_order; j++) {
            sum = sum + fabs(prec_values[j]);
        }
        accur = sqrt(sum);
        if (accur <= eps) break;
	}

	*max_iter = k;
    *prec = accur;
	cudaStatus = cudaMemcpy(x_solution, dev_x_solution, vector_size, cudaMemcpyDeviceToHost);
    error_check(cudaStatus, MEMCPY, dev_a, dev_x_solution, dev_b, dev_prec_values); 

    cudaFree(dev_a);
    cudaFree(dev_x_solution);
    cudaFree(dev_prec_values);
    cudaFree(dev_b);
}

int main(int argc, char *argv[]){

    const int matrix_order = atoi(argv[1]); // order of the matrix
    int max_iter = atoi(argv[2]);     // number of max_iterations
    const float epsilon = atof(argv[3]);    // precision

    int iterations = max_iter;
    std::chrono::time_point<std::chrono::system_clock> start_time, end_time;

    float *x_solution_h, *b_h, **A_h, *rand_x_h;
    float *extended_a = 0;    

    // Allocate memory for CPU.
    A_h = new float *[matrix_order];
    b_h = new float[matrix_order];
    x_solution_h = new float[matrix_order];
    rand_x_h = new float[matrix_order];

    if (A_h == NULL || b_h == NULL || rand_x_h == NULL || x_solution_h == NULL) {
        std::cerr << "Error while allocating resources." << std::endl;
        exit(-1);
    }

    init_matrix(A_h, matrix_order);
    srand(SEED);
    generate_random_matrix(A_h, matrix_order);
    
    extended_a = (float*)malloc(matrix_order*matrix_order*sizeof(float));
	
    for (int i = 0; i < matrix_order; i++) {
		for (int j = 0; j < matrix_order; j++) {
			extended_a[i*matrix_order + j] = A_h[i][j];
		}
	}
    
    generate_random_vector(rand_x_h, matrix_order);

    int repetitions = 20;
    float precision = 1.0, err = 0.0;
    
    matrix_vector_multiplication(b_h, A_h, rand_x_h, matrix_order);
    
    for (int m=0; m<repetitions; m++) {
        
        init_const(x_solution_h, 0.0, matrix_order);
        iterations = max_iter;
        err=0.0;

        start_time = std::chrono::system_clock::now();
        cuda_jacobi_solve(extended_a, x_solution_h, b_h, epsilon, matrix_order, &iterations, &precision);
        end_time = std::chrono::system_clock::now();

        std::cout << delta_time(start_time, end_time).count()  << "\t" ;        
        error_on_computation(x_solution_h, A_h, b_h, matrix_order, &err);
    }

        std::cout << "\t" << iterations << "\t" <<  precision << "\t" <<  err  << std::endl;            

    // Release resources
    for (int i = 0; i < matrix_order; i++)
        delete[] A_h[i];

    delete[] A_h;
    delete[] b_h;
    delete[] rand_x_h;
    delete[] x_solution_h;
    free(extended_a);

    return 0;

}