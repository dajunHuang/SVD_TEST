CU_FILES = $(shell find ./ -name '*.cu')
H_FILES = $(shell find ./ -name '*.h')

OBJ_FILES = $(CU_FILES:%.cu=%)

all:  $(OBJ_FILES)

%: %.cu
	nvcc -arch=native -g -l cusolver -l cublas -l curand -I . $< -o $@

clean:
	-rm $(OBJ_FILES)