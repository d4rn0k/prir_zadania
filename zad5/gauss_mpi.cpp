#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "mpi.h"

#include "gauss_mpi.h"
#include "handle_error_macro.h"

#include <limits>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdexcept>
#include <iomanip>
#include <string>
#include <math.h>


#define MASTER 0

#define MSG_FROM_WORKER 666
#define MSG_FROM_MASTER 111

void writeMatToFile(cv::Mat& m, const char* filename){
	std::ofstream fout(filename);

	if(!fout) {
		std::cout<<"File Not Opened"<<std::endl;
		return;
	}

	std::setfill(" ");
	for(int i=0; i<m.rows; i++) {
		for(int j=0; j<m.cols; j++) {

			cv::Vec3b val = m.at<cv::Vec3b>(i,j);

			fout <<  "["
					<< std::setw(3) << (unsigned int)val.val[0] << ", "
					<< std::setw(3) << (unsigned int)val.val[1] << ", "
					<< std::setw(3) << (unsigned int)val.val[2]
					                                         << "],   ";
		}
		fout<<std::endl;
	}

	fout.close();
}



// Przeciążony szablonowy operator << do printowania dowolnego wektora!
template<typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
	out << "[";
	size_t last = v.size() - 1;
	for(size_t i = 0; i < v.size(); ++i) {
		out << v[i];
		if (i != last)
			out << ", ";
	}
	out << "]";
	return out;
}

// Extern rank, potrzebny do makra HANDLE_ERROR
int rank;

int main(int argc, char *argv[]) {

	int threadsTotal;

	int rowsToCalculate, totalColumns;

	MPI_Status status;
	MPI::Init(argc, argv);

	threadsTotal = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();


	//Rank==0, master proces wczytuje obrazek
	if (rank == MASTER) {

		cv::Mat input_imageMat, output_imageMat;
		std::string outputImagePath;


		int imin = std::numeric_limits<int>::min();
		int imax = std::numeric_limits<int>::max();

		//		fprintf(stdout, "min int value= %d, max int value= %d\n", imin, imax);





		try {

			if (argc != 3) {
				throw std::runtime_error("Błędna liczba parametrów!");
			}

			input_imageMat = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
			outputImagePath = argv[2];

			if (threadsTotal < 0 ) {
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
		if (threadsTotal == 1) {

		}

		output_imageMat = cv::Mat(input_imageMat.rows - 4, input_imageMat.cols - 4, input_imageMat.type());

		// Wysyłamy ilość wierszy, ilość kolumn
		totalColumns = input_imageMat.cols;

		/*
		 * Algorytm podziału:
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
		 * W sumie na wyjściu 31 + 33 + 32 = 96 wierszy bo 100 - 4 = 96 czyli się zgadza! :)
		 *
		 */

		// Ustawiamy tablicę z ilością wierszy na proces
		std::vector<int> rowsPerProcess = getRowsPerProcessDistribution(input_imageMat.rows, threadsTotal);


		writeMatToFile(input_imageMat, "input_image.data");

		fprintf(stdout, "Obrazek wejściowy: wierszy x kolumn: [%d][%d]\n", input_imageMat.rows, input_imageMat.cols);


		std::cout << "Podział: \n";
		std::cout << rowsPerProcess << std::endl;

		//std::cout << "Podział dla 100 elementow na 3 procesy: \n";
		//std::cout << getRowsPerProcessDistribution(100, 3) << std::endl;

		// offset początkowy ustawiony na ilość elementów dla procesu MASTER (0)
		int toSendOffset = (getRowsCountForProcess(input_imageMat.rows, threadsTotal, MASTER) - 2) * totalColumns * 3;


		for (int destination = 1; destination < threadsTotal; destination++) {
			//			rowsToCalculate = getRowsCountForProcess(input_imageMat.rows, threadsTotal, destination) + 4;
			//
			//			if (destination == threadsTotal - 1) {
			//				// Ostatnia porcja danych jest mniejsza o 2 wiersze.
			//				rowsToCalculate -= 2;
			//			}

			int totalPixelsDataCount = rowsPerProcess[destination] * totalColumns * input_imageMat.channels();



			HANDLE_ERROR(MPI_Send(&totalColumns                    , 1                   , MPI_INT,           destination, MSG_FROM_MASTER, MPI_COMM_WORLD));
			HANDLE_ERROR(MPI_Send(&rowsPerProcess[destination], 1                   , MPI_INT,           destination, MSG_FROM_MASTER, MPI_COMM_WORLD));
			HANDLE_ERROR(MPI_Send(&input_imageMat.data[toSendOffset] , totalPixelsDataCount, MPI_UNSIGNED_CHAR, destination, MSG_FROM_MASTER, MPI_COMM_WORLD));

			// Odejmujemy od offsetu 2 wiersze (te poniżej)!
			toSendOffset = toSendOffset + totalPixelsDataCount - (2 * (totalColumns * input_imageMat.channels() ));
		}
		fprintf(stdout, "Wysłano wszystkie dane do początkowe procesów z procesu MASTER\n");


		cv::Mat inputMatrixForFirstProcess = cv::Mat(rowsPerProcess[0], totalColumns, input_imageMat.type());
		cv::Mat outputMatrixForFirstProcess = cv::Mat(inputMatrixForFirstProcess.rows - 4, inputMatrixForFirstProcess.cols - 4, input_imageMat.type());

		memcpy(&inputMatrixForFirstProcess.data[0], &input_imageMat.data[0], inputMatrixForFirstProcess.rows * inputMatrixForFirstProcess.cols * input_imageMat.channels());
		// Proces 0 oblicza swoją część
		do5GaussMPI(&inputMatrixForFirstProcess, &outputMatrixForFirstProcess);

		memcpy(&output_imageMat.data[0], &outputMatrixForFirstProcess.data[0], outputMatrixForFirstProcess.rows * outputMatrixForFirstProcess.cols * input_imageMat.channels());

		// Odbieramy obliczone dane od procesów
		int outputImageOffset = (rowsPerProcess[0] - 4) * (totalColumns - 4) * input_imageMat.channels();
		for (int source = 1; source < threadsTotal; source++) {

			int totalRecvDataCount = (rowsPerProcess[source] - 4) * (totalColumns - 4) * input_imageMat.channels();

			HANDLE_ERROR(MPI_Recv(&output_imageMat.data[outputImageOffset], totalRecvDataCount, MPI_UNSIGNED_CHAR, source, MSG_FROM_WORKER, MPI_COMM_WORLD, &status));

			outputImageOffset += totalRecvDataCount;
			printf("Koniec odbierania danych od: %d\n", source);
		}



		writeMatToFile(output_imageMat, "output_image.data");

		try {
			// Zapis obrazka wyjściowego na dysk
			cv::imwrite(outputImagePath, output_imageMat);

		} catch (std::exception& exc) {
			std::cout << "Błąd podczas zapisu wynikowego obrazka:\n" << exc.what() << std::endl;
		}


		std::cout << "Obrazek zapisany!" << std::endl;


	} else {
		// Rank != 0, każdy inny proces


		// Odbieranie danych początkowych
		HANDLE_ERROR(MPI_Recv(&totalColumns   , 1	, MPI_INT,   MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status));
		HANDLE_ERROR(MPI_Recv(&rowsToCalculate, 1	, MPI_INT,   MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status));



		cv::Mat inputMatrixForProcess= cv::Mat(rowsToCalculate, totalColumns, CV_8UC3);
		// Tworzymy nową macierz na wynikowe dane
		cv::Mat outputMatrixForProcess = cv::Mat(rowsToCalculate - 4, totalColumns - 4, CV_8UC3);

		HANDLE_ERROR(MPI_Recv(&inputMatrixForProcess.data[0], rowsToCalculate * totalColumns * 3	, MPI_UNSIGNED_CHAR,   MASTER, MSG_FROM_MASTER, MPI_COMM_WORLD, &status));

		fprintf(stdout, "Proces %d odebrał %d wierszy od MASTERA i zrobi z tego %d wierszy wynikowych\n", rank, rowsToCalculate , rowsToCalculate - 4);


		std::string fileName = std::to_string(rank) + "inputMatrixForProcess_image.data";
		writeMatToFile(inputMatrixForProcess, fileName.c_str() );



		// Obliczanie swojej części
		do5GaussMPI(&inputMatrixForProcess, &outputMatrixForProcess);

		// Wysyłanie do MASTERA obliczeń
		fprintf(stdout, "Proces %d wysyła dane czyli [%d]x[%d]x[%d]= %d do MASTERa\n",
				rank,
				outputMatrixForProcess.rows,
				outputMatrixForProcess.cols,
				3,
				outputMatrixForProcess.rows * outputMatrixForProcess.cols * 3);

		HANDLE_ERROR(MPI_Send(&outputMatrixForProcess.data[0], outputMatrixForProcess.rows * outputMatrixForProcess.cols * 3,
				MPI_UNSIGNED_CHAR, MASTER, MSG_FROM_WORKER, MPI_COMM_WORLD));

	}



	fprintf(stdout, "Wyczyszczono pamięć!, proces: %d\n", rank);
	MPI::Finalize();
	return 0;

}

std::vector<int> getRowsPerProcessDistribution(int totalRows, int totalProcesses) {
	std::vector<int> rowsPerProcess(totalProcesses);

	rowsPerProcess[0] = getRowsCountForProcess(totalRows, totalProcesses, 0) + 2;
	for (int destination = 1; destination < totalProcesses; destination++) {
		int my_rowsToCalculate = getRowsCountForProcess(totalRows, totalProcesses, destination) + 4;

		if (destination == totalProcesses - 1) {
			// Ostatnia porcja danych jest mniejsza o 2 wiersze.
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

	printf("Wierszy: %d, dla procesu %d / %d\n", rows, currentProcessCount, totalProcesses);

	return rows;
}


void do5GaussMPI(cv::Mat *my_input_img, cv::Mat *my_output_image) {

	int channels = my_input_img->channels();

	uchar* minus2Row;
	uchar* minus1Row;
	uchar* currentRow;
	uchar* plus1Row;
	uchar* plus2Row;

	int row;
	int col;
	int i;


	// Liczniki do potrzeby zapisu w macierzy wynikowej, zaczynają się od zera.
	int output_row = 0;
	int output_col = 0;

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
				//				printf("blueColIndex: %d \n", blueColIndex);

				blueTotal  += minus2Row[blueColIndex ] + minus1Row[blueColIndex ] + currentRow[blueColIndex ] + plus1Row[blueColIndex ] + plus2Row[blueColIndex ];
				greenTotal += minus2Row[greenColIndex] + minus1Row[greenColIndex] + currentRow[greenColIndex] + plus1Row[greenColIndex] + plus2Row[greenColIndex];
				redTotal   += minus2Row[redColIndex  ] + minus1Row[redColIndex  ] + currentRow[redColIndex  ] + plus1Row[redColIndex  ] + plus2Row[redColIndex  ];

			}

			//printf("[%3d][%3d] blueTotal= %d\n", row, col, blueTotal);

			// Zaokrąglamy i robimy niejawną konwersję uchar -> int
			blueTotal  = round(blueTotal  / 25.0f);
			greenTotal = round(greenTotal / 25.0f);
			redTotal   = round(redTotal   / 25.0f);

			//Zapisanie pixela do obrazka wynikowego

			my_output_image->at<cv::Vec3b>(row - 2, col -2) = cv::Vec3b(blueTotal, greenTotal, redTotal);

			//			my_output_image->data[myOutputPointer++] = blueTotal;
			//			my_output_image->data[myOutputPointer++] = greenTotal;
			//			my_output_image->data[myOutputPointer++] = redTotal;

			//			my_output_image->at<cv::Vec3b>(output_row, output_col) = cv::Vec3b(blueTotal, greenTotal, redTotal);
			//			output_col++;
			//			if (col == my_input_img->cols - 3) {
			//				output_col = 0;
			//				output_row++;
			//			}
		}
	}

}
