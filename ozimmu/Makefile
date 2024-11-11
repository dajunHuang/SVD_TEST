CU_FILES = $(shell find ./ -name '*.cu')

all:  ozimmu_test_double

ozimmu_test_double: $(CU_FILES)
	nvcc -arch=native -l cusolver -l cublas -g -I . $(CU_FILES) -o $@

clean:
	-rm ozimmu_test_double