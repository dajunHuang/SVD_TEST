CU_FILES = $(shell find ./ -name '*.cu')
H_FILES = $(shell find ./ -name '*.h')

OBJ_FILES = $(CU_FILES:%.cu=%)

all:  $(OBJ_FILES)

%: %.cu
	nvcc -gencode arch=compute_80,code=sm_80 \
	     -gencode arch=compute_89,code=sm_89 \
	     -gencode arch=compute_90,code=sm_90 \
	     -g -lcublas -lcurand -I . $< -o $@


clean:
	-rm $(OBJ_FILES)