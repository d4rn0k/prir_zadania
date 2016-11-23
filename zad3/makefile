PROGRAM_NAME=macierz_gpu
CC=nvcc
CFLAGS=-std=c++11 -O3 -gencode arch=compute_20,code=sm_20 --compiler-options -Wall

all:
	$(CC) -o $(PROGRAM_NAME) $(PROGRAM_NAME).cu $(CFLAGS)
clean:
	rm $(PROGRAM_NAME)


#nvcc -gencode arch=compute_20,code=sm_20 --compiler-options -Wall -O3 macierz_gpu.cu -o macierz_gpu
