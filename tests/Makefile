include ../make.inc

QUDA = ../lib/libquda.a

INC += -I../include -I.

HDRS = blas_reference.h dslash_reference.h staggered_dslash_reference.h    \
	test_util.h

TESTS = su3_test blas_test pack_test $(WILSON_DIRAC_TEST) 		\
	$(STAGGERED_DIRAC_TEST) $(DOMAIN_WALL_DIRAC_TEST)		\
	$(FATLINK_TEST) $(GAUGE_FORCE_TEST) $(FERMION_FORCE_TEST)

all: $(TESTS)

staggered_dslash_test: staggered_dslash_test.o test_util.o staggered_dslash_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

staggered_invert_test: staggered_invert_test.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

wilson_dslash_test: wilson_dslash_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

wilson_invert_test: wilson_invert_test.o test_util.o dslash_reference.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

domain_wall_dslash_test: domain_wall_dslash_test.o test_util.o domain_wall_dslash_reference.o $(QUDA)

su3_test: su3_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

pack_test: pack_test.o test_util.o dslash_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

blas_test: blas_test.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@

llfat_test: llfat_test.o llfat_reference.o test_util.o misc.o $(QUDA)
	$(CC) $(LDFLAGS) $^  -o $@ 

gauge_force_test: gauge_force_test.o gauge_force_reference.o test_util.o misc.o $(QUDA)
	$(CC) $(LDFLAGS) $^  -o $@ 

fermion_force_test: fermion_force_test.o fermion_force_reference.o test_util.o misc.o $(QUDA)
	$(CC) $(LDFLAGS) $^  -o $@ 


tune: blas_test
	./blas_test $(DEVICE) && cp -f blas_param.h ../lib/

clean:
	-rm -f *.o $(TESTS)

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all tune clean


