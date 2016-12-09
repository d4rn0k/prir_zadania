PROGRAM_NAME = gauss_mpi

CC = mpicxx
CFLAGS = -Wall -std=c++11 -O3

SRCS = src/$(PROGRAM_NAME).cpp

OPENCV = `pkg-config opencv --cflags --libs`
LIBS = $(OPENCV)

all:
	$(CC) $(CFLAGS) -o $(PROGRAM_NAME) $(SRCS) $(LIBS)
clean:
	rm $(PROGRAM_NAME)
