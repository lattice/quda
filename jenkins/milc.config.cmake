# preload file for jenkins test

# MILC - turns on staggered dirac and all HISQ and gauge features for MILC RHMC


set(QUDA_DIRAC_WILSON OFF CACHE BOOL "build Wilson Dirac operators")
set(QUDA_DIRAC_CLOVER OFF CACHE BOOL "build clover Dirac operators")
set(QUDA_DIRAC_DOMAIN_WALL OFF CACHE BOOL "build domain wall Dirac operators")
set(QUDA_DIRAC_STAGGERED ON CACHE BOOL "build staggered Dirac operators")
set(QUDA_DIRAC_TWISTED_MASS OFF CACHE BOOL "build twisted mass Dirac operators")
set(QUDA_DIRAC_TWISTED_CLOVER OFF CACHE BOOL "build twisted clover Dirac operators")
set(QUDA_DIRAC_NDEG_TWISTED_MASS OFF CACHE BOOL "build non-degenerate twisted mass Dirac operators")
set(QUDA_LINK_ASQTAD OFF CACHE BOOL "build code for computing asqtad fat links")
set(QUDA_LINK_HISQ ON CACHE BOOL "build code for computing hisq fat links")
set(QUDA_FORCE_GAUGE ON CACHE BOOL "build code for (1-loop Symanzik) gauge force")
set(QUDA_FORCE_ASQTAD OFF CACHE BOOL "build code for asqtad fermion force")
set(QUDA_FORCE_HISQ ON CACHE BOOL "build code for hisq fermion force")
set(QUDA_GAUGE_TOOLS OFF CACHE BOOL "build auxiliary gauge-field tools")
set(QUDA_GAUGE_ALG OFF CACHE BOOL "build gauge-fixing and pure-gauge algorithms")
set(QUDA_CONTRACT OFF CACHE BOOL "build code for bilinear contraction")
set(QUDA_DYNAMIC_CLOVER OFF CACHE BOOL "Dynamically invert the clover term for twisted-clover")
set(QUDA_QIO OFF CACHE BOOL "build QIO code for binary I/O")


set(QUDA_MULTIGRID OFF CACHE BOOL "build multigrid solvers")