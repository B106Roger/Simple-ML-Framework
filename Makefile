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

# -march=skylake-avx512 for avx512
# -mavx for mm
# -funroll-loops for unroll loop
CXXFLAGS = -mavx -fno-tree-vectorize -O3 -Wall -Wl,--no-as-needed -shared -std=c++11 -fPIC -I${INCLUDE_DIRS} $(MKL_LIBS) -ldl -lpthread -lm $(INCLUDE) -I$(INCLUDE_CORE)

MATLIB = _matrix${shell python3-config --extension-suffix}

.PHONY: all
all: ${MATLIB}

${MATLIB}: ./core/pybindwrapper.cpp ./core/matrix_simd_256.cpp ./core/base_layer.cpp ./core/linear.cpp  ./core/network.cpp ./core/loss.cpp ./core/optimizer.cpp
	${CXX} ${FPIC} $? -o $@ ${CXXFLAGS} 
	python -c "import _matrix"
#   cp $@ ./testcase/$@
	

test: ${MATLIB}
	python -m unittest main_matrix_test.py -v
	python -m unittest main_performance_test.py -v
#	python -m unittest main_layer_test.py
#	python -m unittest main_mnist_test.py


clean:
	rm -rf *.so __pycache__ .pytest_cache 