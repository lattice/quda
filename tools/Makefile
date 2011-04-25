include ../make.inc

CUDA_INC_DIR=${CUDA_INSTALL_PATH}/include
CUDA_LIB_DIR=${CUDA_INSTALL_PATH}/lib64
LIBS=-L${CUDA_LIB_DIR} -lnuma -lcudart


gpu_affinity_test: gpu_affinity_test.c
	${FECC} -O3 -I${CUDA_INC_DIR} -o $@ $< ${LIBS}

clean: 
	rm -f gpu_affinity_test *.o
