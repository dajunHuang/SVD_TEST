CU_FILES = $(shell find ./ -name '*.cpp')
H_FILES = $(shell find ./ -name '*.h')

OBJ_FILES = $(CU_FILES:%.cpp=%)

all:  $(OBJ_FILES)

%: %.cpp
	hipcc -l hipblas -I . $< -o $@

clean:
	-rm $(OBJ_FILES)
