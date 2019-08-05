# preload file for jenkins test

# MILC - turns on staggered dirac and all HISQ and gauge features for MILC RHMC


set(QUDA_DIRAC_WILSON ON CACHE BOOL "build Wilson Dirac operators")
set(QUDA_DIRAC_CLOVER ON CACHE BOOL "build clover Dirac operators")
set(QUDA_DIRAC_DOMAIN_WALL OFF CACHE BOOL "build domain wall Dirac operators")
set(QUDA_DIRAC_STAGGERED OFF CACHE BOOL "build staggered Dirac operators")
set(QUDA_DIRAC_TWISTED_MASS OFF CACHE BOOL "build twisted mass Dirac operators")
set(QUDA_DIRAC_TWISTED_CLOVER OFF CACHE BOOL "build twisted clover Dirac operators")
set(QUDA_DIRAC_NDEG_TWISTED_MASS OFF CACHE BOOL "build non-degenerate twisted mass Dirac operators")
set(QUDA_LINK_ASQTAD OFF CACHE BOOL "build code for computing asqtad fat links")
set(QUDA_LINK_HISQ OFF CACHE BOOL "build code for computing hisq fat links")
set(QUDA_FORCE_GAUGE OFF CACHE BOOL "build code for (1-loop Symanzik) gauge force")
set(QUDA_FORCE_ASQTAD OFF CACHE BOOL "build code for asqtad fermion force")
set(QUDA_FORCE_HISQ OFF CACHE BOOL "build code for hisq fermion force")
set(QUDA_GAUGE_TOOLS OFF CACHE BOOL "build auxiliary gauge-field tools")
set(QUDA_GAUGE_ALG OFF CACHE BOOL "build gauge-fixing and pure-gauge algorithms")
set(QUDA_CONTRACT OFF CACHE BOOL "build code for bilinear contraction")
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