include ../make.inc

QUDA = ../lib/libquda.a

INC += -I../include -I.

HDRS = blas_reference.h dslash_reference.h staggered_dslash_reference.h test_util.h

TESTS = su3_test blas_test pack_test dslash_test invert_test staggered_dslash_test staggered_invert_test

all: $(TESTS)

staggered_dslash_test: staggered_dslash_test.o test_util.o staggered_dslash_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

staggered_invert_test: staggered_invert_test.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

dslash_test: dslash_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

invert_test: invert_test.o test_util.o dslash_reference.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

su3_test: su3_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

pack_test: pack_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

blas_test: blas_test.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

colorspinor_test: colorspinor_test.o dslash_reference.o blas_reference.o test_util.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

tune: blas_test
	./blas_test $(DEVICE) && cp -f blas_param.h ../lib/

clean:
	-rm -f *.o $(TESTS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC)  $(NVCCFLAGS) $< -c -o $@

.PHONY: all tune clean


