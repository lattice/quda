include ../make.inc

QUDA = ../lib/libquda.a

INC += -I../include -I.

HDRS = blas_reference.h wilson_dslash_reference.h staggered_dslash_reference.h    \
	domain_wall_dslash_reference.h test_util.h dslash_util.h

TESTS = su3_test blas_test pack_test $(WILSON_DIRAC_TEST) 		\
	$(STAGGERED_DIRAC_TEST) $(DOMAIN_WALL_DIRAC_TEST)		\
	$(FATLINK_TEST) $(GAUGE_FORCE_TEST) $(FERMION_FORCE_TEST) \
	$(UNITARIZE_LINK_TEST) $(HISQ_PATHS_FORCE_TEST) $(HISQ_UNITARIZE_FORCE_TEST)

all: $(TESTS)

staggered_dslash_test: staggered_dslash_test.o test_util.o staggered_dslash_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

staggered_invert_test: staggered_invert_test.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

wilson_dslash_test: wilson_dslash_test.o test_util.o wilson_dslash_reference.o misc.o $(QIO_UTIL) $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

wilson_invert_test: wilson_invert_test.o test_util.o wilson_dslash_reference.o blas_reference.o misc.o $(QIO_UTIL) $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

domain_wall_dslash_test: domain_wall_dslash_test.o test_util.o domain_wall_dslash_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

domain_wall_invert_test: domain_wall_invert_test.o test_util.o domain_wall_dslash_reference.o misc.o \
	blas_reference.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

su3_test: su3_test.o test_util.o wilson_dslash_reference.o misc.o $(QIO_UTIL) $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

pack_test: pack_test.o test_util.o wilson_dslash_reference.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

blas_test: blas_test.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

llfat_test: llfat_test.o llfat_reference.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

gauge_force_test: gauge_force_test.o gauge_force_reference.o test_util.o misc.o $(QUDA)
	$(CC) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

fermion_force_test: fermion_force_test.o fermion_force_reference.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

unitarize_link_test: unitarize_link_test.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

hisq_paths_force_test: hisq_paths_force_test.o hisq_force_reference.o fermion_force_reference.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

hisq_unitarize_force_test: hisq_unitarize_force_test.o hisq_force_reference.o test_util.o misc.o $(QUDA)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

clean:
	-rm -f *.o staggered_dslash_test staggered_invert_test		\
	wilson_dslash_test wilson_invert_test domain_wall_dslash_test	\
	domain_wall_invert_test su3_test pack_test blas_test		\
	llfat_test gauge_force_test fermion_force_test 			\
	hisq_paths_force_test hisq_unitarize_force_test			\
	unitarize_links_test	

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all clean
