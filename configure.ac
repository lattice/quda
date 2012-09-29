AC_INIT([quda],[0.4],[quda-developers@googlegroups.com])


dnl Specify CUDA Location
AC_ARG_WITH(cuda,
  AC_HELP_STRING([--with-cuda=CUDADIR], [ Specify CUDA installation directory (default: /usr/local/cuda)]),
  [ cuda_home=${withval} ],
  [ cuda_home=/usr/local/cuda ]
)

dnl Specify CPU Arch
AC_ARG_ENABLE(cpu-arch,
 AC_HELP_STRING([--enable-cpu-arch=arch], [ Set CPU arch: x86 or x86_64 (default: x86_64)]),
 [ cpu_arch=${enableval} ],
 [ cpu_arch="x86_64" ]
)

dnl Specify GPU Arch
AC_ARG_ENABLE(gpu-arch,
 AC_HELP_STRING([--enable-gpu-arch=arch], [ Set GPU arch: sm_11, sm_12, sm_13, sm_20, sm_21, sm_30, sm_35 (default: sm_20)]),
 [ gpu_arch=${enableval} ],
 [ gpu_arch="sm_20" ]
)

AC_ARG_ENABLE(os, 
  AC_HELP_STRING([--enable-os=os], [ Set operating system: linux, osx (default: linux)]),
  [ quda_os=${enableval} ], 
  [ quda_os="linux" ]
)

AC_ARG_WITH(python, 
  AC_HELP_STRING([--with-python=path ], [ Set python executable used for 'make gen']),
  [ quda_python=${withval} ],
  [ quda_python="python" ]
)

AC_ARG_ENABLE(host-debug, 
  AC_HELP_STRING([--enable-host-debug], [ Enable debugging of host code]),
  [ quda_host_debug=${enableval} ], 
  [ quda_host_debug="no" ]
)

AC_ARG_ENABLE(device-debug, 
  AC_HELP_STRING([--enable-device-debug], [ Enable debugging for device code]),
  [ quda_device_debug=${enableval} ],
  [ quda_device_debug="no" ]
)

AC_ARG_ENABLE(verbose-build, 
   AC_HELP_STRING([--enable-verbose-build], [ Display kernel register usage]),
   [ quda_verbose_build=${enableval} ], 
   [ quda_verbose_build="no" ]
)

AC_ARG_ENABLE(dslash-profiling, 
  AC_HELP_STRING([--enable-dslash-profiling], [ Enable profiling of multi-GPU Dslash]),
  [ quda_dslash_profiling=${enableval} ], 
  [ quda_dslash_profiling="no" ]
)

AC_ARG_ENABLE(wilson-dirac,
  AC_HELP_STRING([--enable-wilson-dirac], [ Build Wilson Dirac operator (default: enabled)]),
  [ build_wilson_dirac=${enableval} ], 
  [ build_wilson_dirac="yes" ]
)

AC_ARG_ENABLE(clover-dirac,
  AC_HELP_STRING([--enable-clover-dirac], [ Build Wilson-clover Dirac operator (default: enabled)]),
  [ build_clover_dirac=${enableval} ], 
  [ build_clover_dirac="yes" ]
)

AC_ARG_ENABLE(domain-wall-dirac,
  AC_HELP_STRING([--enable-domain-wall-dirac], [ Build domain wall Dirac operator (default: enabled)]),
  [ build_dwf_dirac=${enableval} ], 
  [ build_dwf_dirac="yes" ]
)

AC_ARG_ENABLE(staggered-dirac,
  AC_HELP_STRING([--enable-staggered-dirac], [ Build staggered Dirac operator (default: enabled)]),
  [ build_staggered_dirac=${enableval} ], 
  [ build_staggered_dirac="yes" ]
)

AC_ARG_ENABLE(twisted-mass-dirac,
  AC_HELP_STRING([--enable-twisted-mass-dirac], [ Build twisted mass Dirac operator (default: enabled)]),
  [ build_twm_dirac=${enableval} ], 
  [ build_twm_dirac="yes" ]
)

AC_ARG_ENABLE(staggered-fatlink,
  AC_HELP_STRING([--enable-staggered-fatlink], [ Build asqtad link-fattening routines (default: disabled)]),
  [ build_staggered_fatlink=${enableval} ], 
  [ build_staggered_fatlink="no" ]
)

AC_ARG_ENABLE(hisq-fatlink,
  AC_HELP_STRING([--enable-hisq-fatlink], [ Build HISQ link-fattening routines (default: disabled)]),
  [ build_hisq_fatlink=${enableval} ], 
  [ build_hisq_fatlink="no" ]
)

AC_ARG_ENABLE(gauge-force,
  AC_HELP_STRING([--enable-gauge-force], [ Build 1-loop Symanzik gauge force (default: disabled)]),
  [ build_gauge_force=${enableval} ], 
  [ build_gauge_force="no" ]
)

AC_ARG_ENABLE(staggered-force,
  AC_HELP_STRING([--enable-staggered-force], [ Build asqtad fermion force (default: disabled)]),
  [ build_staggered_force=${enableval} ], 
  [ build_staggered_force="no" ]
)

AC_ARG_ENABLE(hisq-force,
  AC_HELP_STRING([--enable-hisq-force], [ Build HISQ fermion force (default: disabled)]),
  [ build_hisq_force=${enableval} ],
  [ build_hisq_force="no" ]
)
dnl Multi-GPU options
dnl ==================

dnl enable multi-gpu mode
AC_ARG_ENABLE(multi-gpu, 
  AC_HELP_STRING([--enable-multi-gpu], [ Enable Multi-GPU support (default: disabled)]),
  [ multi_gpu=${enableval} ],
  [ multi_gpu="no" ]
)

dnl enable comms compute/overlap
AC_ARG_ENABLE(overlap-comms,
  AC_HELP_STRING([--enable-overlap-comms], [ Enable comms/compute overlap (default: enabled)]),
  [ overlap_comms=${enableval}],
  [ overlap_comms="yes" ]
)

dnl enable cuda / nic interop buffer
AC_ARG_ENABLE(gpu-direct,
  AC_HELP_STRING([--enable-gpu-direct], [ Enable CUDA/NIC interop pinned memory (default: enabled)]),
  [ gpu_direct=${enableval}],
  [ gpu_direct="yes" ]
)

dnl enable packing and unpacking on the device
AC_ARG_ENABLE(device-pack,
  AC_HELP_STRING([--enable-device-pack], [ Enable the packing / unpacking of fields on the device (default: disabled)]),
  [ device_pack=${enableval}],
  [ device_pack="no" ]
)

AC_ARG_WITH(mpi,
 AC_HELP_STRING([--with-mpi=MPIDIR], [ Specify MPI installation directory]),
 [ mpi_home=${withval}; build_mpi="yes"],
 [ mpi_home=""; build_mpi="no" ]
)


AS_CASE([$build_mpi],
	[yes], [AC_PROG_CC($mpi_home/bin/mpicc mpicc) AC_PROG_CXX($mpi_home/bin/mpicxx mpicxx CC)],
	[no], [AC_PROG_CC(gcc cc) AC_PROG_CXX(g++ CC cxx)])


AC_ARG_WITH(qmp,
 AC_HELP_STRING([--with-qmp=QMPDIR], [ Specify QMP installation directory]),
 [ qmp_home=${withval} ; build_qmp="yes" ],
 [ qmp_home="" ; build_qmp="no" ]
)

AC_ARG_WITH(qio,
 AC_HELP_STRING([--with-qio=QIODIR], [ Specify QIO installation directory]),
 [ qio_home=${withval} ; build_qio="yes" ],
 [ qio_home="" ; build_qio="no" ]
)

AC_ARG_ENABLE(qdp-jit,
  AC_HELP_STRING([--enable-qdp-jit], [ Enable QDP-JIT support, requires --with-qdp (default: disabled)]),
  [ build_qdpjit=${enableval} ], 
  [ build_qdpjit="no" ]
)

AC_ARG_WITH(qdp,
  AC_HELP_STRING([--with-qdp=QDPDIR], [ Specify QDP++ installation directory]),
  [ qdp_home=${withval} ],
  [ qdp_home="" ]
)

if test "X${build_qdpjit}X" = "XyesX" ; then
  if test "X${qdp_home}X" = "XX" ; then
    AC_PATH_PROG(QDPXX_CONFIG, [qdp++-config], [])
  else
    AC_PATH_PROG(QDPXX_CONFIG, [qdp++-config], [], [${qdp_home}/bin:${PATH}])
  fi

  if test "X${QDPXX_CONFIG}X" = "XX" ; then
    AC_MSG_ERROR([QDP++ configuration program qdp++-config not found.])
  fi

  dnl AC_MSG_NOTICE([Found QDP++ configuration program ${QDPXX_CONFIG}])
  dnl AC_SUBST(CXX, "`${QDPXX_CONFIG} --cxx`")
  dnl AC_MSG_NOTICE([QDP++ compiler: ${CXX}])
  AC_SUBST(QDPXX_CXXFLAGS, "`${QDPXX_CONFIG} --cxxflags`")
  AC_MSG_NOTICE([QDP++ compile flags: ${QDPXX_CXXFLAGS}])
  AC_SUBST(QDPXX_LDFLAGS,  "`${QDPXX_CONFIG} --ldflags`")
  AC_MSG_NOTICE([QDP++ linking flags: ${QDPXX_LDFLAGS}])
  AC_SUBST(QDPXX_LIBS,     "`${QDPXX_CONFIG} --libs`")
  AC_MSG_NOTICE([QDP++ libraries flags: ${QDPXX_LIBS}])
fi



AC_ARG_ENABLE(fermi-double-tex,
 AC_HELP_STRING([--enable-fermi-double-tex], [ Enable double-precision texture reads on Fermi (default: enabled)]),
 [ fermi_double_tex=${enableval}],
 [ fermi_double_tex="yes" ]
)

AC_ARG_ENABLE(numa-affinity,
 AC_HELP_STRING([--enable-numa-affinity], [ Enable NUMA affinity support (default: enabled, always disabled on osx target)]),
 [ numa_affinity=${enableval}],
 [ numa_affinity="yes" ]
)
dnl Input validation

dnl CPU Arch
case ${cpu_arch} in
x86 | x86_64 ) ;;
*) 
  AC_MSG_ERROR([ CPU arch must be either x86_64 or x86 ])
  ;;
esac

dnl CPU Arch
case ${gpu_arch} in
sm_11 | sm_12 | sm_13 | sm_20 | sm_21 | sm_30 | sm_35) ;;
*) 
  AC_MSG_ERROR([ GPU arch must be one of: sm_11, sm_12, sm_13, sm_20, sm_21, sm_30, or sm_35 ])
  ;;
esac

dnl OS
case ${quda_os} in
linux|osx);;
*) 
  AC_MSG_ERROR([ --enable-os only supports 'linux' and 'osx' ])
  ;;
esac

dnl HOST DEBUG
case ${quda_host_debug} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-host-debug ])
  ;;
esac

dnl DEVICE DEBUG
case ${quda_device_debug} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for  --enable-device-debug  ])
  ;;
esac

dnl VERBOSE
case ${quda_verbose_build} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-quda-verbose-build ])
  ;;
esac

dnl DSLASH_PROFILING
case ${quda_dslash_profiling} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-dslash-profiling ])
  ;;
esac

dnl Build Wilson Dirac
case ${build_wilson_dirac} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-wilson-dirac : ${build_wilson_dirac} ])
  ;;
esac

dnl Build Clover Dirac
case ${build_clover_dirac} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-clover-dirac : ${build_clover_dirac} ])
  ;;
esac

dnl Build DWF Dirac
case ${build_dwf_dirac} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-domain-wall-dirac ])
  ;;
esac

dnl Build Staggered Dirac
case ${build_staggered_dirac} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-staggered-dirac ])
  ;;
esac

dnl Build Twisted Mass Dirac
case ${build_twm_dirac} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-twisted-mass-dirac ])
  ;;
esac

dnl Build Staggered fatlink
case ${build_staggered_fatlink} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-staggered-fatlink ])
  ;;
esac

dnl Build Hisq fatlink
case ${build_hisq_fatlink} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-hisq-fatlink ])
  ;;
esac

dnl Build Gauge Force
case ${build_gauge_force} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-gauge-force ])
  ;;
esac

dnl Build Staggered force
case ${build_staggered_force} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-staggered-force ])
  ;;
esac

dnl Build Hisq force
case ${build_hisq_force} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-hisq-force ])
  ;;
esac

dnl Multi GPU
case ${multi_gpu} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-multi-gpu ])
  ;;
esac

dnl Enables overlapping comms
case ${overlap_comms} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-overlap-comms ])
  ;;
esac

dnl Enables CUDA/NIC buffer interop
case ${gpu_direct} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --gpu-direct ])
  ;;
esac

dnl QDP-JIT support
case ${build_qdpjit} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-qdp-jit : ${build_qdpjit} ])
  ;;
esac


dnl Configuration phase
dnl ===================
dnl
dnl Currently DWF doesnt support parallelism -- that needs a different structure
dnl so if Multi-GPU is turned on I should disable this

dnl QMP/MPI: It is legal to have both --with-qmp and --with-mpi
dnl defined together. This means that: Use QMP with the MPI in MPIHOME.
dnl but we have to disable mpi_comms

if test "X${build_qmp}X" = "XyesX";
then 
  build_mpi="no"
fi
   
if test "X${multi_gpu}X" = "XyesX";
then 
  AC_MSG_NOTICE([Enabling Multi-GPU])
  
  AC_MSG_NOTICE([Asqtad fermion force doesn't support multi-GPU yet: disabling ])
  build_staggered_force="no";


  if test "X${qmp_home}X" = "XX"; then
#    if test "X${mpi_home}X" = "XX"; then
    if test "X${build_mpi}X" = "XnoX"; then
        AC_MSG_WARN([ Multi-GPU build without QMP or MPI.  Will build single node code with copies ])
    fi
  else
 
    dnl qmp home is not empty 
    if test "X${mpi_home}X" = "XX"; 
    then 
      AC_MSG_WARN([ It looks like QMP is being used, but --with-mpi is NOT set... BEWARE ])
    fi
  fi 
fi

if test "X${build_qio}X" = "XyesX"; then   
  if test "X${build_qmp}X" = "XnoX"; then
    AC_MSG_ERROR([QMP must enabled for QIO support ])
  fi	
fi
 
dnl Enables textures on Fermi
case ${fermi_double_tex} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-fermi-double-tex ])
  ;;
esac

case ${numa_affinity} in
yes|no);;
*)
  AC_MSG_ERROR([ invalid value for --enable-numa-affinity ])
  ;;
esac

dnl Output Substitutions
AC_MSG_NOTICE([Setting CUDA_INSTALL_PATH = ${cuda_home} ])
AC_SUBST( CUDA_INSTALL_PATH, [${cuda_home} ])

AC_MSG_NOTICE([Setting CPU_ARCH = ${cpu_arch} ])
AC_SUBST( CPU_ARCH, [${cpu_arch}] )

AC_MSG_NOTICE([Setting GPU_ARCH = ${gpu_arch} ])
AC_SUBST( GPU_ARCH, [${gpu_arch}] )

AC_MSG_NOTICE([Setting OS = ${quda_os} ])
AC_SUBST( QUDA_OS,  [${quda_os}]  )

AC_MSG_NOTICE([Setting QUDA_PYTHON = ${quda_python} ])
AC_SUBST( QUDA_PYTHON,   [${quda_python}] )

AC_MSG_NOTICE([Setting HOST_DEBUG = ${quda_host_debug} ])
AC_SUBST( HOST_DEBUG,    [${quda_host_debug}] )

AC_MSG_NOTICE([Setting DEVICE_DEBUG = ${quda_device_debug} ])
AC_SUBST( DEVICE_DEBUG,  [${quda_device_debug}] )

AC_MSG_NOTICE([Setting VERBOSE = ${quda_verbose_build} ])
AC_SUBST( VERBOSE,	 [${quda_verbose_build}] )

AC_MSG_NOTICE([Setting DSLASH_PROFILING = ${quda_dslash_profiling} ])
AC_SUBST( DSLASH_PROFILING,	 [${quda_dslash_profiling}] )

AC_MSG_NOTICE([Setting BUILD_WILSON_DIRAC = ${build_wilson_dirac} ])
AC_SUBST( BUILD_WILSON_DIRAC, [${build_wilson_dirac}])

AC_MSG_NOTICE([Setting BUILD_CLOVER_DIRAC = ${build_clover_dirac} ])
AC_SUBST( BUILD_CLOVER_DIRAC, [${build_clover_dirac}])

AC_MSG_NOTICE([Setting BUILD_DOMAIN_WALL_DIRAC = ${build_dwf_dirac} ])
AC_SUBST( BUILD_DOMAIN_WALL_DIRAC, [${build_dwf_dirac}])

AC_MSG_NOTICE([Setting BUILD_STAGGERED_DIRAC = ${build_staggered_dirac} ])
AC_SUBST( BUILD_STAGGERED_DIRAC, [${build_staggered_dirac}])

AC_MSG_NOTICE([Setting BUILD_TWISTED_MASS_DIRAC = ${build_twm_dirac} ])
AC_SUBST( BUILD_TWISTED_MASS_DIRAC, [${build_twm_dirac}])

AC_MSG_NOTICE([Setting BUILD_FATLINK = ${build_staggered_fatlink} ] )
AC_SUBST( BUILD_FATLINK, [${build_staggered_fatlink}])

AC_MSG_NOTICE([Setting BUILD_HISQLINK = ${build_hisq_fatlink} ] )
AC_SUBST( BUILD_HISQLINK, [${build_hisq_fatlink}])

AC_MSG_NOTICE([Setting BUILD_GAUGE_FORCE = ${build_gauge_force} ] )
AC_SUBST( BUILD_GAUGE_FORCE, [${build_gauge_force}])

AC_MSG_NOTICE([Setting BUILD_FERMION_FORCE = ${build_staggered_force} ] )
AC_SUBST( BUILD_FERMION_FORCE, [${build_staggered_force}])

AC_MSG_NOTICE([Setting BUILD_HISQ_FORCE = ${build_hisq_force} ] )
AC_SUBST( BUILD_HISQ_FORCE, [${build_hisq_force}])

AC_MSG_NOTICE([Setting BUILD_MULTI_GPU = ${multi_gpu} ] )
AC_SUBST( BUILD_MULTI_GPU, [${multi_gpu}])

AC_MSG_NOTICE([Setting BUILD_QMP = ${build_qmp} ])
AC_SUBST( BUILD_QMP, [${build_qmp}])

AC_MSG_NOTICE([Setting BUILD_MPI = ${build_mpi} ])
AC_SUBST( BUILD_MPI, [${build_mpi}])

AC_MSG_NOTICE([Setting OVERLAP_COMMS= ${overlap_comms}])
AC_SUBST( OVERLAP_COMMS, [${overlap_comms}])

AC_MSG_NOTICE([Setting GPU_DIRECT= ${gpu_direct}])
AC_SUBST( GPU_DIRECT, [${gpu_direct}])

AC_MSG_NOTICE([Setting DEVICE_PACK= ${device_pack}])
AC_SUBST( DEVICE_PACK, [${device_pack}])

AC_MSG_NOTICE([Setting BUILD_QIO = ${build_qio} ])
AC_SUBST( BUILD_QIO, [${build_qio}])

AC_MSG_NOTICE([Setting FECC = ${CC} ])
AC_SUBST( FECC, [${CC}])

AC_MSG_NOTICE([Setting FECXX = ${CXX} ])
AC_SUBST( FECXX, [${CXX}])

AC_MSG_NOTICE([Setting FEF90 = ${F90} ])
AC_SUBST( FEF90, [${F90}])

AC_MSG_NOTICE([Setting MPI_HOME=${mpi_home}])
AC_SUBST( MPI_HOME, [${mpi_home}] )

AC_MSG_NOTICE([Setting QMP_HOME=${qmp_home}])
AC_SUBST( QMP_HOME, [${qmp_home}] )

AC_MSG_NOTICE([Setting QIO_HOME=${qio_home}])
AC_SUBST( QIO_HOME, [${qio_home}] )

AC_MSG_NOTICE([Setting FERMI_DBLE_TEX= ${fermi_double_tex}])
AC_SUBST( FERMI_DBLE_TEX, [${fermi_double_tex}])

AC_MSG_NOTICE([Setting NUMA_AFFINITY= ${numa_affinity}])
AC_SUBST( NUMA_AFFINITY, [${numa_affinity}])

AC_MSG_NOTICE([Setting USE_QDPJIT = ${build_qdpjit} ])
AC_SUBST( USE_QDPJIT, [${build_qdpjit}])

AC_MSG_NOTICE([Setting QDP_INSTALL_PATH = ${qdp_home} ])
AC_SUBST( QDP_INSTALL_PATH, [${qdp_home} ])

AC_CONFIG_FILES(make.inc)
AC_OUTPUT
