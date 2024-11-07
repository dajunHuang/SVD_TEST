CU_FILES = $(shell find ./ -name '*.cu')
H_FILES = $(shell find ./ -name '*.h')

OBJ_FILES = $(CU_FILES:%.cu=%)

%: %.cu
	nvcc -arch=native -l cusolver -l cublas -I . $< -o $@

all:  $(OBJ_FILES)

clean:
	-rm $(OBJ_FILES)