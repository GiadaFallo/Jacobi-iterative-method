ICC = icc
GPP = g++-4.8

CXXFLAGS = -std=c++11 -pthread -O3 
REPORTFLAG = #-qopt-report-phase=vec -qopt-report=2
MIC = -mmic

# Location of the CUDA Toolkit
CUDA_PATH = "/usr/local/cuda-8.0"
NVCC = $(CUDA_PATH)/bin/nvcc -std=c++11
CUDAFLAGS = -I $(CUDA_PATH)/include

src = src/main.cpp src/jacobi_solver.cpp src/jacobi_seq_solver.cpp src/jacobi_thread_barrier_solver.cpp src/jacobi_ff_solver.cpp 
 
FFLOCAL = -I /home/spm1501/public/fastflow -DNO_DEFAULT_MAPPINGS

jacobi_xeon:
	$(ICC) $(src) $(CXXFLAGS) -o bin/$@ $(FFLOCAL) $(REPORTFLAG)

jacobi_titanic:
	$(GPP) $(src) $(CXXFLAGS) -o bin/$@ -I /home/spm1601/fastflow $(REPORTFLAG) -DNO_DEFAULT_MAPPINGS

jacobi_mic:
	$(ICC) $(src) $(MIC) $(CXXFLAGS) -o bin/$@ $(FFLOCAL) $(REPORTFLAG)
	scp bin/jacobi_mic mic0:bin/jacobi_mic

jacobi_cuda:
		$(NVCC) $(CUDAFLAGS) -Wno-deprecated-gpu-targets -o bin/jacobi_cuda src/jacobi_cuda.cu