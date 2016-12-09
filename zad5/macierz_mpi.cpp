#include <iostream>
#include <chrono>
#include <stdexcept>
#include <string>
#include <math.h>
#include "mpi.h"

#define MASTER 0

#define MSG_FROM_WORKER 666
#define MSG_FROM_MASTER 111


// Dla ciągłej alokacji macierzy!
float** alloc_2d_float(int rows, int cols) {
    float *data = (float *)malloc(rows*cols*sizeof(float));
    float **array= (float **)malloc(rows*sizeof(float*));

    for (int i=0; i<rows; i++) {
    	array[i] = &(data[cols*i]);
    }

    return array;
}

void printMatrix(float **matrix, int rows, int columns){
	for (int i = 0; i < rows; i++)
	{
		printf("\n");
		for (int j = 0; j < columns; j++)
			printf("%6.f   ", matrix[i][j]);
	}
}

int getElemsCountForProcess(int matrixSize, int totalProcesses, int procCount) {

	procCount++;

	int rows = matrixSize / totalProcesses;

	if (procCount == totalProcesses) {
		rows = rows + matrixSize % totalProcesses;
	}

	return rows * matrixSize;
}

void multiplyPartOfMatrix(float **matA, float **matB, float **matC, int matrixSize, int rowsOffset, int elemsToCount) {

	int rowsCount = elemsToCount / matrixSize;

//	printf("rowsOffset: %d, rowsCount: %d matrixSize: %d, elemsToCount: %d\n",
//			rowsOffset,
//			rowsCount,
//			matrixSize,
//			elemsToCount
//	);

	for (int ii = rowsOffset; ii < rowsCount; ii++) {
		for (int kk = 0; kk < matrixSize; kk++) {
			for (int jj = 0; jj < matrixSize; jj++) {
				matC[ii][jj] += matA[ii][kk] * matB[kk][jj];
			}
		}
	}
	printf("Policzono!\n");
}

int main(int argc, char *argv[]) {

	int rank, elemsForProc, totalProcesses, matrixSize;
	int rowsOffset = 0;

	double starttime, endtime;

	MPI_Status status;
	matrixSize = sqrt(std::stoi(argv[1]));

	MPI::Init(argc, argv);
	totalProcesses = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();

	if (argc != 2 || std::stoi(argv[1]) < 0 ) {

		std::cout
		<< "Nieprawidlowa liczba parametrow lub bledne parametry!\n\nPrawidlowe wywolanie: "
		<< argv[0] << " <matrixSize>\nGdzie matrixSize = Rozmiar macierzy.\n\n";

		MPI_Abort(MPI_COMM_WORLD, -1);
		MPI::Finalize();
	}

	// SQRT


	float **A;
	float **B;
	float **C;

	A = alloc_2d_float(matrixSize, matrixSize);
	B = alloc_2d_float(matrixSize, matrixSize);
	C = alloc_2d_float(matrixSize, matrixSize);

	// Proces glowny
	if (rank == MASTER) {

		try {
			for (int i = 0; i < matrixSize; i++){
				//Wypelnianie macierzy
				for (int j = 0; j < matrixSize; j++){
					A[i][j] = fmod( (float)(sinf((float)i) *  i * j) , (float)10);
					B[i][j] = fmod( (float)(cosf((float)j) * (i + j)), (float)10);
					C[i][j] = 0;
				}
			}
		}
		catch (std::bad_alloc & exc) {
			printf("Błąd alokacji!");
			MPI::Finalize();
			return -2;
		}

		if (totalProcesses != 1) {

			starttime = MPI_Wtime();

			int rowsOffset2 = getElemsCountForProcess(matrixSize, totalProcesses, 0) / matrixSize;
			// Wysyłamy: macierz A, B, oraz przypowiadajace im ilosci elementow do reszty procesow
			for(int destination = 1; destination < totalProcesses; destination++){

				elemsForProc = getElemsCountForProcess(matrixSize, totalProcesses, destination);
				printMatrix(B, matrixSize, matrixSize);
				MPI_Send(&elemsForProc,                   1,   MPI_INT, destination, MSG_FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&rowsOffset2,                    1,   MPI_INT, destination, MSG_FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&A[rowsOffset2][0],   elemsForProc, MPI_FLOAT, destination, MSG_FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&B[0][0], matrixSize * matrixSize , MPI_FLOAT, destination, MSG_FROM_MASTER, MPI_COMM_WORLD);

				rowsOffset2 += elemsForProc / matrixSize;
			}

			// Proces MASTER zaczyna liczyc swoja czesc danych
				multiplyPartOfMatrix(A, B, C, matrixSize, 0, getElemsCountForProcess(matrixSize, totalProcesses, MASTER));

			// Odbieranie wynikow od procesow
			for (int source = 1; source < totalProcesses; source++) {

				MPI_Recv(&elemsForProc,                1, MPI_INT,    source, MSG_FROM_WORKER, MPI_COMM_WORLD, &status);
				printf("Odebrano elemsForProc: %d z procesu: %d \n", elemsForProc, source);
				MPI_Recv(&rowsOffset,                  1, MPI_INT,    source, MSG_FROM_WORKER, MPI_COMM_WORLD, &status);
				printf("Odebrano rowsOffset %d z procesu: %d \n", rowsOffset, source);
				MPI_Recv(&C[rowsOffset][0], elemsForProc, MPI_DOUBLE, source, MSG_FROM_WORKER, MPI_COMM_WORLD, &status);
				printf("Koniec odbierania danych od: %d\n", source);
			}

			endtime   = MPI_Wtime();

			printf("Czas %.0f ms\n", (endtime - starttime) * 1000);
		} else {
			// Przypadek gdy mamy dostepny tylko jeden proces

			starttime = MPI_Wtime();
				multiplyPartOfMatrix(A, B, C, matrixSize, 0, matrixSize * matrixSize );
			endtime   = MPI_Wtime();

			//Do testu poprawnosci obliczen (porownanie z lab1 na openMP)

			printf("Jeden Proces A[5][5]:= %f\n", A[5][5]);
			printf("Jeden Proces B[5][5]:= %f\n", B[5][5]);
			printf("Jeden Proces C[5][5]:= %f\n", C[5][5]);

			printf("Jeden Proces A[999][999]:= %f\n", A[999][999]);
			printf("Jeden Proces B[999][999]:= %f\n", B[999][999]);
			printf("Jeden Proces C[999][999]:= %f\n", C[999][999]);

			printf("Czas %.0f ms\n", (endtime - starttime)*1000);
		}
	} else {

		// Procesy robocze (rank != MASTER) odbieraja dane
		MPI_Recv(&elemsForProc,               1	, MPI_INT,   MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&rowsOffset,                 1 , MPI_INT,   MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&A[rowsOffset][0], elemsForProc, MPI_FLOAT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);


		MPI_Recv(&B[0][0], matrixSize * matrixSize, MPI_FLOAT, MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status);
		printf("Proces %d odebrał macierz B\n", rank);


		// Licza swoje czesci macierzy...
		multiplyPartOfMatrix(A, B, C, matrixSize, rowsOffset, elemsForProc);

		// I wysylaja do MASTER'a
		MPI_Send(&elemsForProc,            	   1, MPI_INT,   MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD);
		MPI_Send(&rowsOffset, 				   1, MPI_INT,   MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD);
		MPI_Send(&C[rowsOffset][0], elemsForProc, MPI_FLOAT, MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD);
//		printf("Koniec wysyłania danych z procesu: %d\n", rank);
	}

	// Czyszczenie pamieci
	free(A);
	free(B);
	free(C);
	printf("Wyczyszczono pamięć!, proces: %d\n", rank);
	MPI::Finalize();
	return 0;
}
