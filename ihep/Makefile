include ../make.inc

QUDA = ../lib/libquda.a


INC += -I ../include -I ../tests -I .

IHEP_PATH = ./
TESTS_PATH = ../tests/

HDRS = blas_reference.h wilson_dslash_reference.h staggered_dslash_reference.h    \
	domain_wall_dslash_reference.h test_util.h dslash_util.h $(IHEP_PATH)ihep_utility.h

F_RES = $(IHEP_PATH)ihep_three_point.o $(IHEP_PATH)ihep_two_point.o $(IHEP_PATH)ihep_two_point_B.o\
	$(IHEP_PATH)res_set_gamma.o $(IHEP_PATH)res_set_geo.o $(IHEP_PATH)ker_loop.o\
	$(IHEP_PATH)res_su3_mult.o
C_RES = $(IHEP_PATH)ihep_clover.o $(IHEP_PATH)ihep_stout_smear.o $(IHEP_PATH)ihep_utility.o $(IHEP_PATH)ihep_clover.o\
        $(IHEP_PATH)ihep_Jacobi_smear.o
all: ihep_inverter

ihep_inverter: $(IHEP_PATH)ihep_invert.o $(TESTS_PATH)test_util.o $(TESTS_PATH)wilson_dslash_reference.o\
               $(TESTS_PATH)domain_wall_dslash_reference.o $(TESTS_PATH)blas_reference.o $(TESTS_PATH)misc.o \
	       $(C_RES) $(F_RES) $(QIO_UTIL) $(QUDA)  
	       $(CXX) $(LDFLAGS) $^ -o $@ $(LDFLAGS) -limf -lm -lifcore 


clean:
	-rm -f *.o ihep_inverter

%.o: %.c $(HDRS)
	$(CC) $(CFLAGS) $< -c -o $@

%.o: %.cpp $(HDRS)
	$(CXX) $(CXXFLAGS) $< -c -o $@

%.o: %.f90 
	$(F90) $(FTNFLAGS) $< -c -o $@

%.o: %.f
	$(F90) $(FTNFLAGS) $< -c -o $@

%.o: %.cu $(HDRS)
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

.PHONY: all clean
