CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3 
hw2a: CFLAGS += -pthread -ftree-vectorize -fopt-info-vec-all -march=native
hw2b: CC = mpicc 
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -ftree-vectorize -fopt-info-vec-all -march=native


hw2b_opt1: CC = mpicc
hw2b_opt1: CXX = mpicxx
hw2b_opt1: CFLAGS += -fopenmp -ftree-vectorize
hw2b_opt2: CC = mpicc
hw2b_opt2: CXX = mpicxx
hw2b_opt2: CFLAGS += -fopenmp -ftree-vectorize

hw2_pthread: CFLAGS += -pthread

CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2_pthread hw2a hw2b hw2b_opt1 hw2b_opt2

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
