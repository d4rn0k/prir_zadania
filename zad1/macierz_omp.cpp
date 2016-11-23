
#include <math.h>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>

void checkMatrix(float** cpu_A, int matrixSize) {

	int indexes[] = { 0, 5, 10, matrixSize / 2, matrixSize - 1 };

	for (size_t i = 0; i < sizeof(indexes) / sizeof(int); i ++){
		int mIndex = indexes[i];
		printf("Wartość dev_C[%d][%d] = %f\n", mIndex, mIndex, cpu_A[mIndex][mIndex]);
	}
}

int main(int argc, char *argv[])
{
	int matrixSize;
	int threadCount;

	int i,j,k;

	float **A, **B, **C;

	if (argc != 3 || std::stoi(argv[2]) < 0 ) {
		std::cout
			<< "Nieprawidlowa liczba parametrow lub bledne parametry! Prawidlowe wywolanie: "
			<< argv[0]
			<< " <n> <size>\n\nGdzie \n\nn = Liczba watkow\nSize = Rozmiar macierzy.\n\n";
		std::cout << "argc size=" << argc;
		return -1;
	}


	threadCount = std::stoi(argv[1]);
	matrixSize = sqrt(std::stoi(argv[2]));

	try {

		//Inicjalizacja macierzy
		A = new float*[matrixSize];
		B = new float*[matrixSize];
		C = new float*[matrixSize];

		for (int i = 0; i < matrixSize; i++){
			A[i] = new float[matrixSize];
			B[i] = new float[matrixSize];
			C[i] = new float[matrixSize];
		}

	}
	catch (std::bad_alloc & exc) {
		throw std::invalid_argument("Bład alokacji! Za duzy rozmiar macierzy / za malo pamieci!");
	}

	//Wypełnianie macierzy
	for (int i = 0; i < matrixSize; i++){
		for (int j = 0; j < matrixSize; j++){
			A[i][j] = fmod( (float)(sinf((float)i) *  i * j) , (float)10);
			B[i][j] = fmod((float)(cosf((float)j) * (i + j)), (float)10);
			C[i][j] = 0;
		}
	}

	//Główna pętla programu
	auto start = std::chrono::system_clock::now();
	
	#pragma omp parallel for private(i, j, k) firstprivate(matrixSize) num_threads(threadCount)
	for ( i = 0; i < matrixSize; i++){
		for ( k = 0; k < matrixSize; k++){
			for ( j = 0; j < matrixSize; j++){
				C[i][j] += A[i][k] * B[k][j];
			}
		}
	}

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start);
	std::cout << "Czas: " << duration.count() << "ms\n";

	checkMatrix(C, matrixSize);
	
	//Czyszczenie pamięci
	for (int i = 0; i < matrixSize; i++) {
		delete[] A[i];
		delete[] B[i];
		delete[] C[i];
	}
	delete[]A;
	delete[]B;
	delete[]C;

	return 0;
}

