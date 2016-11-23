#include "macierz_gpu.h"

#include <iostream>
#include <math.h>

int BLOCK_SIZE;

__global__ void matrixMultiplyKernel(Matrix A, Matrix B, Matrix C) {

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	int matrixSize = A.matrixSize;

	float tempCValue = 0.0f;

	//	printf("row(%d) = threadIdx.y(%d) + blockIdx.y(%d) * blockDim.y(%d),  col(%d) = threadIdx.x(%d) + blockIdx.x(%d) * blockDim.x(%d)\n",
	//			row, threadIdx.y, blockIdx.y, blockDim.y, col, threadIdx.x, blockIdx.x, blockDim.x);

	if (row < matrixSize && col < matrixSize){

		for(int k = 0; k < matrixSize; k++) {
			tempCValue += GetElement(A, row, k) * GetElement(B, k, col);
		}

		//__syncthreads();
		SetElement(C, row, col, tempCValue);
	}
}

void matrixMultiplyAndGenerateHost(Matrix cpu_A, Matrix cpu_B, Matrix cpu_C) {

	float elapsedTime;
	Matrix dev_A, dev_B, dev_C;
	cudaEvent_t startTime, stopTime;
	size_t totalSize = cpu_A.matrixSize * cpu_A.matrixSize * sizeof(float);

	//	printf("Alokujemy %lu B, %lu KB pamieci, na %lu  elementow\n", totalSize, totalSize / 1024, totalSize / sizeof(float) );

	// Deklarujemy  siatkę i wątki na blok
	dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

	dim3 grid( (int) ceil(cpu_A.matrixSize/ (float)threadsPerBlock.x),(int) ceil(cpu_A.matrixSize / (float)threadsPerBlock.y));

	dev_A.matrixSize = dev_B.matrixSize = dev_C.matrixSize = cpu_A.matrixSize;

	//Alokacja macierzy dev_A, dev_B, dev_C
	CUDA_CHECK_RETURN( cudaMalloc(&dev_A.elements, totalSize) );
	CUDA_CHECK_RETURN( cudaMalloc(&dev_B.elements, totalSize) );
	CUDA_CHECK_RETURN( cudaMalloc(&dev_C.elements, totalSize) );

	// Generowanie macierzy!
	generateMatrixes(dev_A, dev_B, dev_A.matrixSize);

	// Jeśli wymiar macierzy == liczba bloków
	if (grid.x == 1) {
		grid = dim3(1, 1);
	}

	CUDA_CHECK_RETURN( cudaEventCreate ( &startTime));
	CUDA_CHECK_RETURN( cudaEventCreate ( &stopTime));

	// start mierzenia czasu
	CUDA_CHECK_RETURN( cudaEventRecord(startTime, 0) );


	//printf("Uruchamiamy kernel z parametrami: threadsPerBlock.x = %d, threadsPerBlock.y = %d, grid.x = %d, grid.y = %d\n", threadsPerBlock.x, threadsPerBlock.y, grid.x, grid.y);


	// Wywołanie kernela
	matrixMultiplyKernel<<< grid , threadsPerBlock >>> (dev_A, dev_B, dev_C);

	CUDA_CHECK_RETURN( cudaEventRecord(stopTime, 0));
	CUDA_CHECK_RETURN( cudaEventSynchronize(stopTime));
	CUDA_CHECK_RETURN( cudaEventElapsedTime(&elapsedTime, startTime, stopTime));
	printf("Czas wykonania: %.0lf ms\n",elapsedTime);


	//	printf("Uruchamiamy kernel z parametrami: threadsPerBlock.x = %d, threadsPerBlock.y = %d, grid.x = %d, grid.y = %d\n",
	//			threadsPerBlock.x, threadsPerBlock.y, grid.x, grid.y);

	// Kopiowanie macierzy z GPU na CPU
	CUDA_CHECK_RETURN(cudaMemcpy(cpu_C.elements, dev_C.elements, totalSize, cudaMemcpyDeviceToHost));

	// Kopiowanie wygenerowanych macierzy A, B pomocnicze
	//	CUDA_CHECK_RETURN(cudaMemcpy(cpu_A.elements, dev_A.elements, totalSize, cudaMemcpyDeviceToHost));
	//	CUDA_CHECK_RETURN(cudaMemcpy(cpu_B.elements, dev_B.elements, totalSize, cudaMemcpyDeviceToHost));

	//  Sprawdza niektóre elementy macierzy wynikowej cpu_C do testu poprawności!
	//		checkMatrix(cpu_C);


	//Czyszczenie macierzy dev_A, dev_B, dev_C na urządzaniu
	cudaFree(dev_A.elements);
	cudaFree(dev_B.elements);
	cudaFree(dev_C.elements);
}

// Pobiera element z macierzy
__device__ float GetElement(const Matrix A, int row, int col) {
	return A.elements[row * A.matrixSize + col];
}

// Ustawianie konkretnego elementu
__device__ void SetElement(Matrix A, int row, int col, float value) {
	//printf("Ustawiamy element macierzy[%d] = %f\n", row * A.stride + col, value);
	A.elements[row * A.matrixSize + col] = value;
}

__global__ void generateMatrixesKernel(Matrix dev_A, Matrix dev_B, int matrixSize){

	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	if (row < matrixSize && col < matrixSize ){

		float AVal = fmod( (float) ( sinf( (float) row) * (row * col)), 10.0f );
		float BVal = fmod( (float) ( cosf( (float) col) * (row + col)), 10.0f );

		SetElement(dev_A, row, col, AVal);
		SetElement(dev_B, row, col, BVal);
	}

}


// Funkcja zlecająca wypełnianianie macierzy
void generateMatrixes(Matrix A, Matrix B, int matrixSize){

	dim3 threadsPerBlock(32, 32);
	dim3 grid(matrixSize/threadsPerBlock.x, matrixSize/threadsPerBlock.y);

	//kernel<<<blocks, threads>>> (int param1, float param2);
	generateMatrixesKernel<<<grid, threadsPerBlock>>> (A, B, matrixSize);

	CUDA_CHECK_RETURN(cudaThreadSynchronize());
	//printf("Wygenerowano Macierze!\n");
}


int main(int argc, char* argv[]){

	Matrix cpu_A, cpu_B, cpu_C;

	int matrixSize;
	size_t totalSize;

	if (argc != 2 || ( (atoi(argv[1])) <= 0) ) {
		printf("Błąd, złe parametry\n\nWywołanie: %s <wymiar boku macierzy>\n\n", argv[0]);
		exit(-1);
	}

	//CUDA_CHECK_RETURN( cudaSetDevice(1) );
	matrixSize = atoi(argv[1]);
	BLOCK_SIZE = 16;

	totalSize = matrixSize * matrixSize * sizeof(float);

	//	printf("Obliczamy macierze wymiarów: %d x %d\n", matrixSize, matrixSize);

	cpu_A.matrixSize = cpu_B.matrixSize = cpu_C.matrixSize = matrixSize;

	cpu_A.elements = (float*)malloc(totalSize);
	cpu_B.elements = (float*)malloc(totalSize);
	cpu_C.elements = (float*)malloc(totalSize);

	CUDA_CHECK_RETURN(cudaSetDevice(1));
	matrixMultiplyAndGenerateHost(cpu_A, cpu_B, cpu_C);

	//
	//	Matrix cpu_B_after_gen;
	//	cpu_B_after_gen.matrixSize = matrixSize;
	//	cpu_B_after_gen.elements = (float*)malloc(totalSize);
	//	CUDA_CHECK_RETURN(cudaMemcpy(cpu_B_after_gen.elements, cpu_B.elements, totalSize, cudaMemcpyDeviceToHost));

	return 0;
}

void checkMatrix(Matrix cpu_A) {

	int indexes[] = { 0, 5, 10, cpu_A.matrixSize / 2, cpu_A.matrixSize - 1 };

	for (size_t i = 0; i < sizeof(indexes) / sizeof(int); i ++){
		int mIndex = indexes[i];
		printf("Wartość dev_C[%d][%d] = %f\n", mIndex, mIndex, cpu_A.elements[mIndex* cpu_A.matrixSize + mIndex]);
	}
}

void printMatrix(Matrix matrix, int rows, int cols){
	for(int i = 0; i < rows; i++){
		for(int j = 0; j < cols; j++)
			printf("%7.2f ", matrix.elements[i*matrix.matrixSize + j]);
		printf("\n");
	}
	printf("\n");
}

static void checkCudaError (const char *file, unsigned line, const char *statement, cudaError_t err) {
	if (err == cudaSuccess) {
		return;
	}
	std::cerr << statement<<" returned " << cudaGetErrorString(err) << "("<<err<< ") at "<<file<<":"<<line << std::endl;
	exit (1);
}
