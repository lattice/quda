include ../make.inc

QUDA = ../lib/libquda.a

LIME = $(HOME)/install/lime-1.3.2/lib/liblime.a

INC += -I../include -I. -I$(HOME)/install/lime-1.3.2/include

HDRS = blas_reference.h wilson_dslash_reference.h staggered_dslash_reference.h    \
	domain_wall_dslash_reference.h test_util.h dslash_util.h

TESTS = su3_test blas_test dslash_test invert_test $(DIRAC_TEST)			\
	$(FATLINK_TEST) $(GAUGE_FORCE_TEST)	\
	$(FERMION_FORCE_TEST) $(UNITARIZE_LINK_TEST)			\
	$(HISQ_PATHS_FORCE_TEST) $(HISQ_UNITARIZE_FORCE_TEST)

all: $(TESTS)

dslash_test: dslash_test.o test_util.o wilson_dslash_reference.o domain_wall_dslash_reference.o misc.o $(QIO_UTIL) $(QUDA) $(LIME)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

invert_test: invert_test.o test_util.o wilson_dslash_reference.o domain_wall_dslash_reference.o blas_reference.o misc.o $(QIO_UTIL) $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

staggered_invert_test: staggered_invert_test.o test_util.o staggered_dslash_reference.o misc.o blas_reference.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

su3_test: su3_test.o test_util.o misc.o $(QIO_UTIL) $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

blas_test: blas_test.o test_util.o misc.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

llfat_test: llfat_test.o llfat_reference.o test_util.o misc.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

gauge_force_test: gauge_force_test.o gauge_force_reference.o test_util.o misc.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

fermion_force_test: fermion_force_test.o fermion_force_reference.o test_util.o misc.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

unitarize_link_test: unitarize_link_test.o test_util.o misc.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS)

hisq_paths_force_test: hisq_paths_force_test.o hisq_force_reference.o hisq_force_reference2.o fermion_force_reference.o test_util.o misc.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

hisq_unitarize_force_test: hisq_unitarize_force_test.o hisq_force_reference.o test_util.o misc.o $(QUDA)  $(LIME)
	$(CXX) $(LDFLAGS) $^  -o $@  $(LDFLAGS)

clean:
	-rm -f *.o dslash_test invert_test staggered_dslash_test	\
	staggered_invert_test su3_test pack_test blas_test llfat_test	\
	gauge_force_test fermion_force_test hisq_paths_force_test	\
	hisq_unitarize_force_test unitarize_links_test

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all clean
