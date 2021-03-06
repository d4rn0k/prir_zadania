/*
 * gauss_mpi.h
 *
 *  Created on: 5 gru 2016
 *      Author: student
 */

#ifndef GAUSS_MPI_H_
#define GAUSS_MPI_H_


extern int rank;

int getRowsCountForProcess(int totalRows, int totalProcesses, int currentProcessCount);

void do5GaussMPI(cv::Mat *my_input_img, cv::Mat *my_output_image, int rowsToCalculate, int colsToCalculate);

std::vector<int> getRowsPerProcessDistribution(int totalRows, int totalProcesses);


#endif /* GAUSS_MPI_H_ */
