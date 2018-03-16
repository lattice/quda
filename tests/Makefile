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

ifeq ($(strip $(BUILD_FATLINK)), yes)
  FATLINK_TEST=llfat_test
endif

ifeq ($(strip $(BUILD_HISQLINK)), yes)
  ifneq ($(strip $(BUILD_FATLINK)), yes) 
    FATLINK_TEST=llfat_test
  endif
  UNITARIZE_LINK_TEST=unitarize_link_test
endif

ifeq ($(strip $(BUILD_GAUGE_FORCE)), yes)
  GAUGE_FORCE_TEST=gauge_force_test
endif

ifeq ($(strip $(BUILD_FERMION_FORCE)), yes)
  FERMION_FORCE_TEST=fermion_force_test
endif

ifeq ($(strip $(BUILD_HISQ_FORCE)), yes)
  HISQ_PATHS_FORCE_TEST=hisq_paths_force_test
  HISQ_UNITARIZE_FORCE_TEST=hisq_unitarize_force_test
endif

ifeq ($(strip $(BUILD_GAUGE_ALG)), yes)
  GAUGE_ALG_TEST= gauge_alg_test
endif

TESTS = su3_test pack_test blas_test dslash_test invert_test		\
	deflated_invert_test multigrid_invert_test multigrid_benchmark_test $(DIRAC_TEST)	\
	$(STAGGERED_DIRAC_TEST) $(FATLINK_TEST) $(GAUGE_FORCE_TEST)	\
	$(FERMION_FORCE_TEST) $(UNITARIZE_LINK_TEST)			\
	$(HISQ_PATHS_FORCE_TEST) $(HISQ_UNITARIZE_FORCE_TEST)		\
	$(GAUGE_ALG_TEST)

all: $(TESTS)

dslash_test: dslash_test.o test_util.o gtest-all.o wilson_dslash_reference.o clover_reference.o domain_wall_dslash_reference.o blas_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

invert_test: invert_test.o test_util.o wilson_dslash_reference.o clover_reference.o domain_wall_dslash_reference.o blas_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

multigrid_invert_test: multigrid_invert_test.o test_util.o wilson_dslash_reference.o clover_reference.o domain_wall_dslash_reference.o blas_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

multigrid_benchmark_test: multigrid_benchmark_test.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

deflated_invert_test: deflated_invert_test.o test_util.o wilson_dslash_reference.o domain_wall_dslash_reference.o blas_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

staggered_dslash_test: staggered_dslash_test.o gtest-all.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) 

staggered_invert_test: staggered_invert_test.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

su3_test: su3_test.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

gauge_alg_test: gauge_alg_test.o test_util.o misc.o gtest-all.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

pack_test: pack_test.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

blas_test: blas_test.o gtest-all.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

llfat_test: llfat_test.o llfat_reference.o test_util.o misc.o face_gauge.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

gauge_force_test: gauge_force_test.o gauge_force_reference.o test_util.o misc.o face_gauge.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

fermion_force_test: fermion_force_test.o fermion_force_reference.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

unitarize_link_test: unitarize_link_test.o gtest-all.o test_util.o misc.o face_gauge.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

hisq_paths_force_test: hisq_paths_force_test.o hisq_force_reference.o hisq_force_reference2.o fermion_force_reference.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

hisq_unitarize_force_test: hisq_unitarize_force_test.o hisq_force_reference.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

clean:
	-rm -f *.o dslash_test invert_test deflated_invert_test	\
	staggered_dslash_test staggered_invert_test su3_test	\
	pack_test blas_test llfat_test gauge_force_test		\
	fermion_force_test hisq_paths_force_test		\
	hisq_unitarize_force_test unitarize_link_test		\
	multigrid_invert_test multigrid_benchmark_test

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all clean
