# preload file for jenkins test

# MILC - turns on staggered dirac and all HISQ and gauge features for MILC RHMC

set(QUDA_DIRAC_WILSON OFF CACHE BOOL "build Wilson Dirac operators")
set(QUDA_DIRAC_CLOVER OFF CACHE BOOL "build clover Dirac operators")
set(QUDA_DIRAC_DOMAIN_WALL OFF CACHE BOOL "build domain wall Dirac operators")
set(QUDA_DIRAC_STAGGERED ON CACHE BOOL "build staggered Dirac operators")
set(QUDA_DIRAC_TWISTED_MASS OFF CACHE BOOL "build twisted mass Dirac operators")
set(QUDA_DIRAC_TWISTED_CLOVER OFF CACHE BOOL "build twisted clover Dirac operators")
set(QUDA_DYNAMIC_CLOVER OFF CACHE BOOL "Dynamically invert the clover term for twisted-clover")
set(QUDA_QIO OFF CACHE BOOL "build QIO code for binary I/O")

set(QUDA_MULTIGRID OFF CACHE BOOL "build multigrid solvers")
