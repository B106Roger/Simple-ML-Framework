CXX = g++

INCLUDE_DIRS = /home/user/anaconda3/include
# INCLUDE_DIRS = /usr/include/mkl

MKL_DIR = /home/user/anaconda3/lib
# MKL_DIR = /usr/lib/x86_64-linux-gnu

# MKL_LIBS = -lmkl_def -lmkl_avx2 -lmkl_core -lmkl_intel_lp64 -lmkl_sequential
MKL_LIBS = ${MKL_DIR}/libmkl_def.so \
	${MKL_DIR}/libmkl_avx2.so \
	${MKL_DIR}/libmkl_core.so \
	${MKL_DIR}/libmkl_intel_lp64.so \
	${MKL_DIR}/libmkl_sequential.so

FPIC = `python3 -m pybind11 --includes`
INCLUDE = `python3-config --includes`
INCLUDE_CORE = core

CXXFLAGS = -O3 -Wall -Wl,--no-as-needed -shared -std=c++11 -fPIC -I${INCLUDE_DIRS} $(MKL_LIBS) -ldl -lpthread -lm $(INCLUDE) -I$(INCLUDE_CORE)

MATLIB = _matrix${shell python3-config --extension-suffix}

.PHONY: all
all: ${MATLIB}

${MATLIB}: ./core/pybindwrapper.cpp ./core/matrix.cpp ./core/base_layer.cpp ./core/linear.cpp  ./core/network.cpp
	rm -rf *.so __pycache__ .pytest_cache
	${CXX} ${FPIC} $? -o $@ ${CXXFLAGS} 
	python -c "import _matrix"
#   cp $@ ./testcase/$@
	

test: ${MATLIB}
#	python performance_test.py
	python -m unittest main_matrix_test.py
	python -m unittest main_layer_test.py


clean:
	rm -rf *.so __pycache__ .pytest_cache 