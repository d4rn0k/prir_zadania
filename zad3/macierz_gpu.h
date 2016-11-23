#include <stdio.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int matrixSize;
	float* elements;

} Matrix;

// Nagłówki:
__global__ void matrixMultiplyKernel(Matrix A, Matrix B, Matrix C);
__global__ void generateMatrixesKernel(Matrix dev_A, Matrix dev_B, int matrixSize);

__device__ float GetElement(const Matrix A, int row, int col);
__device__ void SetElement(Matrix A, int row, int col, float value);

void generateMatrixes(Matrix A, Matrix B, int matrixSize);
void matrixMultiplyAndGenerateHost(const Matrix dev_A, const Matrix dev_B, Matrix cpu_C);
void checkMatrix(Matrix dev_A);
void printMatrix(Matrix matrix, int rows, int cols);

static void checkCudaError (const char *, unsigned, const char *, cudaError_t);
#define CUDA_CHECK_RETURN(value) checkCudaError(__FILE__,__LINE__, #value, value)




