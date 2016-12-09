PROGRAM_NAME = gauss_mpi

CC = mpicxx
CFLAGS = -g -Wall -std=c++11 -O0

SRCS = $(PROGRAM_NAME).cpp

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

all:
	$(CC) $(CFLAGS) -o $(PROGRAM_NAME) $(SRCS) $(LIBS)
clean:
	rm $(PROGRAM_NAME)
