include ../make.inc

QUDA = ../lib/libquda.a

INC += -I../include -I. 

HDRS = blas_reference.h wilson_dslash_reference.h staggered_dslash_reference.h    \
	domain_wall_dslash_reference.h test_util.h dslash_util.h

ifeq ($(strip $(BUILD_WILSON_DIRAC)), yes)
  DIRAC_TEST = dslash_test invert_test
endif

ifeq ($(strip $(BUILD_DOMAIN_WALL_DIRAC)), yes)
  DIRAC_TEST = dslash_test invert_test
endif

ifeq ($(strip $(BUILD_STAGGERED_DIRAC)), yes)
  STAGGERED_DIRAC_TEST=staggered_dslash_test staggered_invert_test
endif

TESTS = su3_test pack_test dslash_test invert_test		\
	$(STAGGERED_DIRAC_TEST)	\
	$(GAUGE_ALG_TEST)

all: $(TESTS)

dslash_test: dslash_test.o test_util.o gtest-all.o wilson_dslash_reference.o domain_wall_dslash_reference.o blas_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

invert_test: invert_test.o test_util.o wilson_dslash_reference.o domain_wall_dslash_reference.o blas_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

staggered_dslash_test: staggered_dslash_test.o gtest-all.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

staggered_invert_test: staggered_invert_test.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

su3_test: su3_test.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

pack_test: pack_test.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

blas_test: blas_test.o gtest-all.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	-rm -f *.o dslash_test invert_test staggered_dslash_test	\
	staggered_invert_test su3_test pack_test blas_test llfat_test	\
	unitarize_link_test multigrid_invert_test

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all clean
