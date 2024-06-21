# preload file for jenkins test

# MILC - turns on staggered dirac and all HISQ and gauge features for MILC RHMC

set(QUDA_DIRAC_WILSON ON CACHE BOOL "build Wilson Dirac operators")
set(QUDA_DIRAC_CLOVER ON CACHE BOOL "build clover Dirac operators")
set(QUDA_DIRAC_DOMAIN_WALL OFF CACHE BOOL "build domain wall Dirac operators")
set(QUDA_DIRAC_STAGGERED OFF CACHE BOOL "build staggered Dirac operators")
set(QUDA_DIRAC_TWISTED_MASS OFF CACHE BOOL "build twisted mass Dirac operators")
set(QUDA_DIRAC_TWISTED_CLOVER OFF CACHE BOOL "build twisted clover Dirac operators")
set(QUDA_DYNAMIC_CLOVER OFF CACHE BOOL "Dynamically invert the clover term for twisted-clover")
set(QUDA_QIO OFF CACHE BOOL "build QIO code for binary I/O")

# advanced 
set(QUDA_MULTIGRID OFF CACHE BOOL "build multigrid solvers")

# build with MILC interface
set(QUDA_INTERFACE_QDP ON CACHE BOOL "build qdp interface")
set(QUDA_INTERFACE_MILC OFF CACHE BOOL "build milc interface")
set(QUDA_INTERFACE_CPS OFF CACHE BOOL "build cps interface")
set(QUDA_INTERFACE_QDPJIT OFF CACHE BOOL "build qdpjit interface")
set(QUDA_INTERFACE_BQCD ON CACHE BOOL "build bqcd interface")
set(QUDA_INTERFACE_TIFR OFF CACHE BOOL "build tifr interface")
