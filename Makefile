#CUDA_HEADER_DIR=/usr/local/cuda-10.2/include
CUDA_HEADER_DIR=/usr/local/cuda-12.5/include
CXX_FLAGS=-Wall -D__CL_ENABLE_EXCEPTIONS

all: langtons_ant

langtons_ant: langtons_ant.cpp
	g++ $(CXX_FLAGS) -I$(CUDA_HEADER_DIR) langtons_ant.cpp -o langtons_ant -g -lOpenCL -lglut -lGLEW -lGLU -lGL -fopenmp

clean:
	rm -rf langtons_ant
