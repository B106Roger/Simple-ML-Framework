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

PYBIND_INC = `python3 -m pybind11 --includes`
PY_INC = `python3-config --includes`

CXXFLAGS = -O3 -Wall -Wl,--no-as-needed -std=c++17 -I${INCLUDE_DIRS} -I./ $(MKL_LIBS) -ldl -lpthread -lm $(INCLUDE)

MATLIB = test # _matrix${shell python3-config --extension-suffix}

.PHONY: all
all: ${MATLIB}

${MATLIB}: ./test.cpp
	${CXX} $< -o $@ ${CXXFLAGS} 


