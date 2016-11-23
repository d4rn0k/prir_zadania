#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <typeinfo>
#include <chrono>
#include <exception>
#include <stdexcept>
#include <math.h>

//using namespace std;

// Nagłówki
void generateSampleMatrix(cv::Mat *outputImage);
void showMatrixInfo(cv::Mat *inputMatrix);
void do5Gauss(cv::Mat *my_input_img, cv::Mat *my_output_image, int threads);
void do5GaussOLD(cv::Mat *my_input_img, cv::Mat *my_output_image, int threads);
bool compareTwoMatrixes(cv::Mat *leftMatrix, cv::Mat *rightMatrix);
void testCorrectness(void (*function)(cv::Mat*, cv::Mat*));

int main (int argc, char** argv) {

	cv::Mat input_image;
	int threadsCount;
	std::string outputImagePath;

	try {

		if (argc != 4) {
			throw std::runtime_error("Błędna liczba parametrów!");
		}

		threadsCount = std::stoi(argv[1]);
		input_image = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
		outputImagePath = argv[3];

		if (threadsCount < 0 ) {
			throw std::runtime_error("Liczba wątków nieprawidłowa!");
		}

		if (outputImagePath.empty()) {
			throw std::runtime_error("Pusta ścieżka obrazka wyjściowego!");
		}

	} catch (std::exception &exc){

		std::cout << "Błąd! Nieprawidłowa liczba lub złe parametry!\nPrawidłowe wywołanie:" << std::endl
				<< argv[0] << " <n> <input_image> <output_image>" << std::endl
				<< "gdzie:" << std::endl
				<< "n - Liczba wątków" << std::endl
				<< "input_image  - ścieżka do pliku obrazu wejściowego w formacie JPEG" << std::endl
				<< "output_image - ścieżka do pliku obrazu wyjściowego w formacie JPEG" << std::endl
				<< "\nBłąd: " << exc.what() << std::endl;

		return -1;
	}

	if ( !input_image.data ) {
		std::cout <<  "Nie można otworzyć, lub znaleść obrazu!" << std::endl ;
		return -1;
	}

//	printf("Wczytano obrazek: w[%d px] h[%d px]\nIlość kanałów: %d\nCzy obrazek jest ciągły w pamięci: %s\nCałkowita ilość elementów=%.0f\n",
//			input_image.cols, input_image.rows, input_image.channels(), (input_image.isContinuous()? "tak": "nie"), (float)input_image.cols * input_image.rows );

	//Tworzymy wynikową macierz o takim samym typie jak wejściowa i mniejszą o 2px z każdego boku
	cv::Mat output_image( input_image.rows - 4, input_image.cols - 4, input_image.type() );
	//	cv::Mat output_imageOld( input_image.rows - 4, input_image.cols - 4, input_image.type() );

	// Start mierzenia czasu
	auto startTime = std::chrono::system_clock::now();

	// Wywołanie szybkiej funkcji działającej na wskaźnikach
	do5Gauss(&input_image, &output_image, threadsCount);

	// Koniec mierzenia czasu
	auto endTime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - startTime);
	std::cout << "Czas: " << endTime.count() << "ms\n";

	//	bool isEqual = (cv::sum(output_image != output_imageOld) == cv::Scalar(0,0,0,0));
	//	printf("Czy macierze są identyczne?: %s\n", isEqual ? "Tak" : "Nie");


	//	printf("Test poprawności metody na wskaźnikach: \n");
	//	testCorrectness(do5Gauss);
	//
	//	printf("Test poprawności metody iterującej wszystko po kolei: \n");
	//	testCorrectness(do5GaussOLD);
	//
	//	printf("Macierz wejściowa do testów:\n");
	//	showMatrixInfo(&myInputTestMatrix);

	//	if(!isEqual) {
	//		compareTwoMatrixes(&output_image, &output_imageOld);
	//	}

	try {
		// Zapis obrazka wyjściowego na dysk
		cv::imwrite(outputImagePath, output_image);

	} catch (std::exception& exc) {
		std::cout << "Błąd podczas zapisu wynikowego obrazka:\n" << exc.what() << std::endl;
	}


	std::cout << "Obrazek zapisany!" << std::endl;
	return 0;
}

void do5Gauss(cv::Mat *my_input_img, cv::Mat *my_output_image, int threads) {

	int channels = my_input_img->channels();

	uchar* minus2Row;
	uchar* minus1Row;
	uchar* currentRow;
	uchar* plus1Row;
	uchar* plus2Row;

	int row;
	int col;
	int i;

	//Główna pętla programu
	#pragma omp parallel for private(row, col, i, minus1Row, minus2Row, currentRow, plus1Row, plus2Row) num_threads(threads)
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

			blueTotal  = round(blueTotal  / 25.0f);
			greenTotal = round(greenTotal / 25.0f);
			redTotal   = round(redTotal   / 25.0f);

			//Zapisanie pixela do obrazka wynikowego
			my_output_image->at<cv::Vec3b>(row - 2, col - 2) = cv::Vec3b(blueTotal, greenTotal, redTotal);
		}
	}

}

void do5GaussOLD(cv::Mat *my_input_img, cv::Mat *my_output_image, int threads) {

	for(int row = 2; row < my_input_img->rows - 2; ++row){

		for(int col = 2; col < my_input_img->cols - 2; ++col){

			int blueTotal  = 0;
			int greenTotal = 0;
			int redTotal   = 0;

			for( int rowI = 0; rowI < 5; ++rowI ) {
				for( int colI = 0; colI < 5; ++colI ) {
					cv::Vec3b pixel = my_input_img->at<cv::Vec3b>(row - 2 + rowI, col - 2 + colI);
					blueTotal  += pixel.val[0];
					greenTotal += pixel.val[1];
					redTotal   += pixel.val[2];
				}

			}

			//printf("[%3d][%3d] OLDblueTotal= %d\n", row, col, blueTotal);

			blueTotal  = round(blueTotal  / 25.0f);
			greenTotal = round(greenTotal / 25.0f);
			redTotal   = round(redTotal   / 25.0f);

			my_output_image->at<cv::Vec3b>(row - 2, col - 2) = cv::Vec3b(blueTotal, greenTotal, redTotal);
		}
	}

}

void generateSampleMatrix(cv::Mat *outputImage) {

	for(int row = 0; row < outputImage->rows; ++row){
		for(int col = 0; col < outputImage->cols; ++col){

			outputImage->at<cv::Vec3b>(row, col) = cv::Vec3b(row, col, 66);
		}
	}

}

bool compareTwoMatrixes(cv::Mat *leftMatrix, cv::Mat *rightMatrix){

	if(leftMatrix->rows != rightMatrix->rows) {
		printf("Macierze mają różną liczbę wierszy!\n");
		return false;
	}

	if(leftMatrix->cols != rightMatrix->cols) {
		printf("Macierze mają różną liczbę kolumn!\n");
		return false;
	}

	if(leftMatrix->channels() != rightMatrix->channels()) {
		printf("Macierze mają różną liczbę Kanałów!");
		return false;
	}

	// Sprawdza czy macierze (lewa i prawa) wejściowe są identyczne
	// Jeśli jeden pixel inny printuje go oraz wychodzi z pętli!
	for(int row = 0; row < leftMatrix->rows; row++) {
		for(int col = 0; col < leftMatrix->cols; col++) {

			cv::Vec3b leftMatrixPixel  = leftMatrix ->at<cv::Vec3b>( row , col  );
			cv::Vec3b rightMatrixPixel = rightMatrix->at<cv::Vec3b>( row , col  );

			bool valid = true;

			if(leftMatrixPixel.val[0] != rightMatrixPixel.val[0]){
				printf("[%d][%d] bluePixel [%d] != [%d]", row, col, leftMatrixPixel.val[0], rightMatrixPixel.val[0]);
				valid = false;
			}

			if(leftMatrixPixel.val[1] != rightMatrixPixel.val[1]){
				printf("[%d][%d] greenPixel [%d] != [%d]", row, col, leftMatrixPixel.val[1], rightMatrixPixel.val[1]);
				valid = false;
			}

			if(leftMatrixPixel.val[2] != rightMatrixPixel.val[2]){
				printf("[%d][%d] redPixel [%d] != [%d]", row, col, leftMatrixPixel.val[2], rightMatrixPixel.val[2]);
				valid = false;
			}

			if(!valid) {
				printf("\n");
				return false;
			}

		}
	}

	return true;
}

void showMatrixInfo(cv::Mat *inputMatrix) {
	printf("Wczytano obrazek: w[%d px] h[%d px]\nIlość kanałów: %d\nCzy obrazek jest ciągły w pamięci: %s\n",
			inputMatrix->cols, inputMatrix->rows, inputMatrix->channels(), (inputMatrix->isContinuous()? "tak": "nie"));

	for(int row = 0; row < inputMatrix->rows; ++row){

		for(int col = 0; col < inputMatrix->cols; ++col){

			cv::Vec3b bgrVector = inputMatrix->at<cv::Vec3b>(row, col);

			printf("[%3d, %3d, %3d] ", bgrVector.val[0], bgrVector.val[1], bgrVector.val[2]);
		}
		printf("\n");
	}
}



