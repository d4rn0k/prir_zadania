#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mpi.h"

#include <math.h>

#include <iostream>
#include <iomanip>
#include <stdexcept>

#include "gauss_mpi.h"
#include "handle_error_macro.h"

#define MASTER 0
#define CHANNELS 3

#define MSG_FROM_WORKER 666
#define MSG_FROM_MASTER 111



// Extern rank, potrzebny do makra HANDLE_ERROR
int rank;

int main(int argc, char *argv[]) {

	int processCount;

	int rowsToCalculate, columnsTotal;

	double startTime = 0.0d, stopTime = 0.0d, timeElapsedTotal = 0.0d;

	MPI_Status status;
	MPI::Init(argc, argv);

	processCount = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();


	//Rank == 0, MASTER proces wczytuje obrazek
	if (rank == MASTER) {

		cv::Mat input_imageMat, output_imageMat;
		std::string outputImagePath;

		try {

			if (argc != 3) {
				throw std::runtime_error("Błędna liczba parametrów!");
			}

			input_imageMat = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
			outputImagePath = argv[2];

			if (processCount < 0 ) {
				throw std::runtime_error("Liczba procesów nieprawidłowa!");
			}

			if (outputImagePath.empty()) {
				throw std::runtime_error("Pusta ścieżka obrazka wyjściowego!");
			}

		} catch (std::exception &exc){

			std::cout << "Błąd! Nieprawidłowa liczba lub złe parametry!\nPrawidłowe wywołanie:" << std::endl
					<< argv[0] << " <input_image> <output_image>" << std::endl
					<< "gdzie:" << std::endl
					<< "input_image  - ścieżka do pliku obrazu wejściowego w formacie JPEG" << std::endl
					<< "output_image - ścieżka do pliku obrazu wyjściowego w formacie JPEG" << std::endl
					<< "\nBłąd: " << exc.what() << std::endl;

			MPI_Abort(MPI_COMM_WORLD, -1);
			MPI::Finalize();
			return -1;
		}

		// Przypadek gdy jest tylko jeden proces
		// Powinien on sam zrobić rozmycie gaussa i zapisać obrazek!
		if (processCount == 1) {

		}

		output_imageMat = cv::Mat(input_imageMat.rows - 4, input_imageMat.cols - 4, input_imageMat.type());

		// Wysyłamy ilość wierszy, ilość kolumn
		columnsTotal = input_imageMat.cols;

		/*
		 *
		 * Algorytm podziału:
		 *
		 * getRowsCountForProcess == Wszystkie oprócz ostatniego procesu dostają po równo ilość wierszy,
		 * a ostatni dostaję resztę co pozostała
		 *
		 * Pierwszy proces  dostaje: getRowsCountForProcess + 2 wierszy
		 * Środkowe procesy dostają: getRowsCountForProcess + 4 wierszy z offsetem wierszy ustawionym na -2 od bieżącego
		 * Ostatni  proces  dostaje: getRowsCountForProcess + 2 wierszy z offsetem wierszy ustawionym na -2 od bieżącego
		 *
		 * Przykładowo:
		 * 100 Wierszy (rows = 100)
		 * 3 procesy (threadsTotal = 3)
		 *
		 * proces 0: przesyłamy wiersze: [0 ,  35] => 35 wierszy, na wyjściu 31 wierszy
		 * proces 1: przesyłamy wiersze: [31,  68] => 37 wierszy, na wyjściu 33 wiersze
		 * proces 2: przesyłamy wiersze: [64, 100] => 36 wierszy, na wyjściu 32 wiersze
		 *
		 * W sumie na wyjściu 31 + 33 + 32 = 96 wierszy bo 100 - 4 = 96 czyli się zgadza, hej ho! :)
		 *
		 */

		// Ustawiamy tablicę z ilością wierszy na proces
		// Tu można się zastanowić nad MPI_SendRecv w celu clean code
		std::vector<int> rowsPerProcess = getRowsPerProcessDistribution(input_imageMat.rows, processCount);

		// offset początkowy ustawiony na ilość elementów dla procesu MASTER (0)
		int toSendOffset = (rowsPerProcess[MASTER] - 2) * columnsTotal * 3;


		for (int destination = 1; destination < processCount; destination++) {

			int pixelsDataCount = rowsPerProcess[destination] * columnsTotal * CHANNELS;

			HANDLE_ERROR(MPI_Send(&columnsTotal,
					1,
					MPI_INT,
					destination,
					MSG_FROM_MASTER,
					MPI_COMM_WORLD)
			);
			HANDLE_ERROR(MPI_Send(&rowsPerProcess[destination],
					1,
					MPI_INT,
					destination,
					MSG_FROM_MASTER, MPI_COMM_WORLD)
			);
			HANDLE_ERROR(MPI_Send(&input_imageMat.data[toSendOffset],
					pixelsDataCount,
					MPI_UNSIGNED_CHAR,
					destination,
					MSG_FROM_MASTER,
					MPI_COMM_WORLD)
			);

			// Odejmujemy od offsetu 2 wiersze (te poniżej)!
			toSendOffset = toSendOffset + pixelsDataCount - (2 * (columnsTotal * CHANNELS ));
		}

		// Proces 0 oblicza swoją część:
		cv::Mat inputMatrixForFirstProcess =
				cv::Mat(rowsPerProcess[0],
						columnsTotal,
						input_imageMat.type()
				);

		cv::Mat outputMatrixForFirstProcess =
				cv::Mat(inputMatrixForFirstProcess.rows - 4,
						inputMatrixForFirstProcess.cols - 4,
						input_imageMat.type()
				);

		memcpy(&inputMatrixForFirstProcess.data[0],
				&input_imageMat.data[0],
				inputMatrixForFirstProcess.rows * inputMatrixForFirstProcess.cols * CHANNELS);

		startTime = MPI_Wtime();
		do5GaussMPI(&inputMatrixForFirstProcess, &outputMatrixForFirstProcess);
		stopTime = MPI_Wtime();

		timeElapsedTotal = (stopTime - startTime) * 1000;


		memcpy(&output_imageMat.data[0],
				&outputMatrixForFirstProcess.data[0],
				outputMatrixForFirstProcess.rows * outputMatrixForFirstProcess.cols * CHANNELS);


		// Odbieramy obliczone dane od reszty procesów
		int outputImageOffset = (rowsPerProcess[MASTER] - 4) * (columnsTotal - 4) * CHANNELS;
		for (int source = 1; source < processCount; source++) {

			int totalRecvDataCount = (rowsPerProcess[source] - 4) * (columnsTotal - 4) * CHANNELS;

			HANDLE_ERROR(MPI_Recv(&output_imageMat.data[outputImageOffset],
					totalRecvDataCount,
					MPI_UNSIGNED_CHAR,
					source,
					MSG_FROM_WORKER,
					MPI_COMM_WORLD,
					&status)
			);

			outputImageOffset += totalRecvDataCount;

			//Musimy pobrać czasy wykonania (timeElapsed) z innych procesów
			double timeElapsedForProcess = 0.0d;
			HANDLE_ERROR(MPI_Recv(&timeElapsedForProcess,
					1,
					MPI_DOUBLE,
					source,
					MSG_FROM_WORKER,
					MPI_COMM_WORLD,
					&status)
			);
			timeElapsedTotal += timeElapsedForProcess;

		}




		try {
			// Zapis obrazka wyjściowego na dysk
			cv::imwrite(outputImagePath, output_imageMat);

		} catch (std::exception& exc) {
			std::cout << "Błąd podczas zapisu wynikowego obrazka:\n" << exc.what() << std::endl;
		}

		//Czyścimy wszystkie macierze
		input_imageMat.release();
		output_imageMat.release();
		inputMatrixForFirstProcess.release();
		outputMatrixForFirstProcess.release();

		std::cout << "Czas " << std::fixed << std::setprecision(0)<< timeElapsedTotal << "ms" << std::endl;

	} else {
		// Rank != 0, każdy inny proces

		// Odbieranie danych początkowych
		HANDLE_ERROR(MPI_Recv(&columnsTotal,
				1,
				MPI_INT,
				MASTER,
				MSG_FROM_MASTER,
				MPI_COMM_WORLD,
				&status)
		);
		HANDLE_ERROR(MPI_Recv(&rowsToCalculate,
				1,
				MPI_INT,
				MASTER,
				MSG_FROM_MASTER,
				MPI_COMM_WORLD,
				&status)
		);


		// Tworzymy macierze na wejścię i wyjście mniejsze o 4
		cv::Mat inputMatrixForProcess = cv::Mat(rowsToCalculate, columnsTotal, CV_8UC3);
		cv::Mat outputMatrixForProcess = cv::Mat(rowsToCalculate - 4, columnsTotal - 4, CV_8UC3);

		// Odbieramy dane potrzebne do obliczeń od MASTERA
		HANDLE_ERROR(MPI_Recv(&inputMatrixForProcess.data[0],
				rowsToCalculate * columnsTotal * 3,
				MPI_UNSIGNED_CHAR,
				MASTER,
				MSG_FROM_MASTER,
				MPI_COMM_WORLD, &status)
		);

		// Obliczanie swojej części
		startTime = MPI_Wtime();
		do5GaussMPI(&inputMatrixForProcess, &outputMatrixForProcess);
		stopTime = MPI_Wtime();
		int myTimeElapsed = (stopTime - startTime) * 1000;

		// Wysyłanie do MASTERA wyników
		HANDLE_ERROR(MPI_Send(&outputMatrixForProcess.data[0],
				outputMatrixForProcess.rows * outputMatrixForProcess.cols * CHANNELS,
				MPI_UNSIGNED_CHAR,
				MASTER,
				MSG_FROM_WORKER,
				MPI_COMM_WORLD)
		);

		//Wysyłamy czas wykonania z każdego procesu!
		HANDLE_ERROR(MPI_Send(&myTimeElapsed,
				1,
				MPI_DOUBLE,
				MASTER,
				MSG_FROM_WORKER,
				MPI_COMM_WORLD)
		);


		//Czyszczenie pamięci!
		inputMatrixForProcess.release();
		outputMatrixForProcess.release();
	}

	MPI::Finalize();
	return 0;
}

std::vector<int> getRowsPerProcessDistribution(int totalRows, int totalProcesses) {
	std::vector<int> rowsPerProcess(totalProcesses);

	rowsPerProcess[0] = getRowsCountForProcess(totalRows, totalProcesses, 0) + 2;

	if(totalProcesses == 1) {
		rowsPerProcess[0] = totalRows;
	}

	for (int destination = 1; destination < totalProcesses; destination++) {
		int my_rowsToCalculate = getRowsCountForProcess(totalRows, totalProcesses, destination) + 4;

		if (destination == totalProcesses - 1) {
			// Ostatnia porcja danych jest mniejsza o 2 wiersze (te dolne!).
			my_rowsToCalculate -= 2;
		}

		rowsPerProcess[destination] = my_rowsToCalculate;
	}

	return rowsPerProcess;
}


// Zwraca liczbę wierszy którą potem każdy proces ma przerobić
int getRowsCountForProcess(int totalRows, int totalProcesses, int currentProcessCount) {

	int rows = 0;

	if (totalProcesses == 1) {
		return totalRows;
	}

	rows = totalRows / totalProcesses;

	if (currentProcessCount + 1 == totalProcesses) {
		rows += totalRows % totalProcesses;
	}

	return rows;
}


// Produkuje romzycie Gauss'a dla wejściowej macierzy
void do5GaussMPI(cv::Mat *my_input_img, cv::Mat *my_output_image) {

	int channels = my_input_img->channels();

	// Wskaźniczki na poszczególne wiersze
	uchar* minus2Row;
	uchar* minus1Row;
	uchar* currentRow;
	uchar* plus1Row;
	uchar* plus2Row;

	int row;
	int col;
	int i;

	// Wskaźnik do zapisu w macierzy wynikowej
	int myOutputPointer = 0;

	//Główna pętla programu
	for(row = 2; row < my_input_img->rows - 2; ++row) {

		minus2Row  = my_input_img->ptr<uchar>(row - 2);
		minus1Row  = my_input_img->ptr<uchar>(row - 1);
		currentRow = my_input_img->ptr<uchar>(row    );
		plus1Row   = my_input_img->ptr<uchar>(row + 1);
		plus2Row   = my_input_img->ptr<uchar>(row + 2);

		for(col = 2; col < my_input_img->cols - 2; ++col){

			int blueTotal  = 0;
			int greenTotal = 0;
			int redTotal   = 0;

			for( i = 0; i < 5; ++i ) {

				// Indeks bazowy
				int index = ( (col - 2) * channels) + (channels * i);
				// B, G, R odpowiednio przesunięte o 0, 1, 2
				int blueColIndex  = index + 0;
				int greenColIndex = index + 1;
				int redColIndex   = index + 2;

				blueTotal  +=
						minus2Row  [blueColIndex] +
						minus1Row  [blueColIndex] +
						currentRow [blueColIndex] +
						plus1Row   [blueColIndex] +
						plus2Row   [blueColIndex];

				greenTotal +=
						minus2Row  [greenColIndex] +
						minus1Row  [greenColIndex] +
						currentRow [greenColIndex] +
						plus1Row   [greenColIndex] +
						plus2Row   [greenColIndex];

				redTotal   +=
						minus2Row  [redColIndex] +
						minus1Row  [redColIndex] +
						currentRow [redColIndex] +
						plus1Row   [redColIndex] +
						plus2Row   [redColIndex];

			}

			// Robimy niejawną konwersję uchar -> int, możemy bo napewno się zmieści!
			blueTotal  = blueTotal  / 25.0f;
			greenTotal = greenTotal / 25.0f;
			redTotal   = redTotal   / 25.0f;

			//Zapisanie pixela do obrazka wynikowego

			my_output_image->data[myOutputPointer++] = blueTotal;
			my_output_image->data[myOutputPointer++] = greenTotal;
			my_output_image->data[myOutputPointer++] = redTotal;
		}
	}

}
