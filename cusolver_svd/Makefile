CU_FILES = $(shell find ./ -name '*.cu')
H_FILES = $(shell find ./ -name '*.h')

all:  $(H_FILES)
	nvcc -arch=native -l cusolver -l cublas -I . $(CU_FILES) -o cusolver_gesvd

clean:
	-rm cusolver_gesvd