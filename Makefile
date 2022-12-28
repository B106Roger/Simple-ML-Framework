CXX = g++


###################################
# MKL Library Header File Path
###################################
MKL_INCLUDE_DIRS =  $(CONDA_PREFIX)/include

###################################
# MKL Library LIB Path
###################################
# MKL_LIBS = -lmkl_def -lmkl_avx2 -lmkl_core -lmkl_intel_lp64 -lmkl_sequential
MKL_LIB_DIR = $(CONDA_PREFIX)/lib
MKL_LIBS = ${MKL_LIB_DIR}/libmkl_def.so.2 \
		   ${MKL_LIB_DIR}/libmkl_avx2.so.2 \
		   ${MKL_LIB_DIR}/libmkl_core.so.2 \
		   ${MKL_LIB_DIR}/libmkl_intel_lp64.so.2 \
		   ${MKL_LIB_DIR}/libmkl_sequential.so.2

###################################
# Pybind Compile Prefix
###################################
FPIC = `python3 -m pybind11 --includes`
PYTHON_INCLUDE = `python3-config --includes`

###########################
# User Code Include Path
###########################
INCLUDE_CORE = core

CXXFLAGS = -mavx -fno-tree-vectorize -O3 -Wall -Wl,--no-as-needed -shared -std=c++11 -fPIC -I${MKL_INCLUDE_DIRS} $(MKL_LIBS) -ldl -lpthread -lm $(PYTHON_INCLUDE) -I$(INCLUDE_CORE)

MATLIB = _matrix${shell python3-config --extension-suffix}

.PHONY: all
all: ${MATLIB}

${MATLIB}: ./core/pybindwrapper.cpp ./core/matrix.cpp ./core/base_layer.cpp ./core/linear.cpp  ./core/network.cpp ./core/loss.cpp ./core/optimizer.cpp
	${CXX} ${FPIC} $? -o $@ ${CXXFLAGS} 
	python -c "import _matrix"

test: ${MATLIB}
	python -m unittest main_pp_performance_test.py -v
#	python -m unittest main_matrix_test.py -v
#	python -m unittest main_performance_test.py -v
#	python -m unittest main_layer_test.py
#	python -m unittest main_mnist_test.py
#	python -m unittest main_matrix_test.py

clean:
	rm -rf *.so __pycache__ .pytest_cache 