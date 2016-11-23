PROGRAM_NAME = gauss_omp

CC = g++
CFLAGS = -g -Wall -std=c++11 -O3 -fopenmp

SRCS = $(PROGRAM_NAME).cpp

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

all:
	$(CC) $(CFLAGS) -o $(PROGRAM_NAME) $(SRCS) $(LIBS)
clean:
	rm $(PROGRAM_NAME)
