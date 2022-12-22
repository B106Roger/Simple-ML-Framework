# Simple ML Framework
## Installation
```
https://github.com/B106Roger/Simple-ML-Framework.git
cd Simple-ML-Framework
pip3 install -r requirements.txt
```
## Build Project
```
make
```
*note that if you aren't using conda to manage environment, you should change the below include path in make by yourself !!!*
```=bash
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
```


## Testing Code
```
make test
```

## Final Project Presentation of nsdhw 21au
final presentation: https://docs.google.com/presentation/d/1jQUTzosQuHBJyuUYS2R96sUD37GK-WjS/edit?usp=sharing&ouid=105896444551749782879&rtpof=true&sd=true
## Final Project Presentation of PP 22au