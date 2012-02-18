AC_INIT([quda],[0.4],[mikec@seas.harvard.edu])

AC_PROG_CC(gcc cc)
AC_PROG_CXX(g++ CC cxx)

AC_CHECK_HEADER([numa.h])

dnl Specify CUDA Location
AC_ARG_WITH(cuda,
  AC_HELP_STRING([--with-cuda=CUDADIR], [ Specify CUDA installation directory ]),
  [ cuda_home=${withval} ],
  [ cuda_home=/usr/local/cuda ]
)

dnl Specify CPU Arch
AC_ARG_ENABLE(cpu-arch,
 AC_HELP_STRING([--enable-cpu-arch=arch],  [ Set CPU arch: x86 or x86_64 (default: x86_64)] ),
 [ cpu_arch=${enableval} ],
 [ cpu_arch="x86_64" ]
)

dnl Specify GPU Arch
AC_ARG_ENABLE(gpu-arch,
 AC_HELP_STRING([--enable-gpu-arch=arch],  [ Set GPU arch=sm_10, sm_12, sm_13, sm_20, sm_21 (default: sm_20)] ),
 [ gpu_arch=${enableval} ],
 [ gpu_arch="sm_20" ]
)

dnl Enable GPU Emulation (CUDA 3.0 only)
AC_ARG_ENABLE(gpu-emulation,
 AC_HELP_STRING([--enable-gpu-emulation], [ Enable GPU emulation (CUDA v3.0). (default is 'no' )]),
 [ gpu_emu=${enableval} ], 
 [ gpu_emu="no" ]
)

AC_ARG_ENABLE(os, 
  AC_HELP_STRING([--enable-os=os], [ Set OS Build style: linux, osx (default is linux)]),
  [ quda_os=${enableval} ], 
  [ quda_os="linux" ]
)

AC_ARG_WITH(python, 
  AC_HELP_STRING([--with-python=path ], [ Set Python to use for make gen]),
  [ quda_python=${withval} ],
  [ quda_python="python" ]
)

AC_ARG_ENABLE(tune-device, 
  AC_HELP_STRING([--enable-tune-device=device], [ Set Device to use for make tune ]),
  [ tune_device=${enableval} ], 
  [ tune_device=0 ]
)

AC_ARG_ENABLE(host-debug, 
  AC_HELP_STRING([--enable-host-debug], [ Enable debugging of host code ]),
  [ quda_host_debug=${enableval} ], 
  [ quda_host_debug="no" ]
)

AC_ARG_ENABLE(device-debug, 
  AC_HELP_STRING([--enable-device-debug], [ Enable debugging for device code]),
  [ quda_device_debug=${enableval} ],
  [ quda_device_debug="no" ]
)

AC_ARG_ENABLE(verbose-build, 
   AC_HELP_STRING([--enable-verbose-build], [ Display Kernel Register Usage ]),
   [ quda_verbose_build=${enableval} ], 
   [ quda_verbose_build="no" ]
)

AC_ARG_ENABLE(dslash-profiling, 
  AC_HELP_STRING([--enable-dslash-profiling], [ Enable profiling of multi-GPU dslash ]),
  [ quda_dslash_profiling=${enableval} ], 
  [ quda_dslash_profiling="no" ]
)

AC_ARG_ENABLE(wilson-dirac,
  AC_HELP_STRING([--enable-wilson-dirac], [ Build Wilson Dirac Op  (default=enabled) ]),
  [ build_wilson_dirac=${enableval} ], 
  [ build_wilson_dirac="yes" ]
)

AC_ARG_ENABLE(clover-dirac,
  AC_HELP_STRING([--enable-clover-dirac], [ Build Clover Dirac Op  (default=enabled) ]),
  [ build_clover_dirac=${enableval} ], 
  [ build_clover_dirac="yes" ]
)

AC_ARG_ENABLE(domain-wall-dirac,
  AC_HELP_STRING([--enable-domain-wall-dirac], [ Build Domain Wall Dirac Op  (default=enabled) ]),
  [ build_dwf_dirac=${enableval} ], 
  [ build_dwf_dirac="yes" ]
)

AC_ARG_ENABLE(staggered-dirac,
  AC_HELP_STRING([--enable-staggered-dirac], [ Build Staggered Dirac Op  (default=enabled) ]),
  [ build_staggered_dirac=${enableval} ], 
  [ build_staggered_dirac="yes" ]
)

AC_ARG_ENABLE(twisted-mass-dirac,
  AC_HELP_STRING([--enable-twisted-mass-dirac], [ Build Twisted Mass Dirac Op  (default=enabled) ]),
  [ build_twm_dirac=${enableval} ], 
  [ build_twm_dirac="yes" ]
)

AC_ARG_ENABLE(staggered-fatlink,
  AC_HELP_STRING([--enable-staggered-fatlink], [ Build Staggered Fat Link Code (default=disabled) ]),
  [ build_staggered_fatlink=${enableval} ], 
  [ build_staggered_fatlink="no" ]
)

AC_ARG_ENABLE(hisq-fatlink,
  AC_HELP_STRING([--enable-hisq-fatlink], [ Build Hisq Fat Link Code (default=disabled) ]),
  [ build_hisq_fatlink=${enableval} ], 
  [ build_hisq_fatlink="no" ]
)

AC_ARG_ENABLE(gauge-force,
  AC_HELP_STRING([--enable-gauge-force], [ Build 1 loop Symanzik gauge force (default=disabled) ]),
  [ build_gauge_force=${enableval} ], 
  [ build_gauge_force="no" ]
)

AC_ARG_ENABLE(staggered-force,
  AC_HELP_STRING([--enable-staggered-force], [ Build Staggered Force Term (default=disabled) ]),
  [ build_staggered_force=${enableval} ], 
  [ build_staggered_force="no" ]
)

AC_ARG_ENABLE(hisq-force,
  AC_HELP_STRING([--enable-hisq-force], [ Build Hisq Force Term (default=disabled) ]),
  [ build_hisq_force=${enableval} ],
  [ build_hisq_force="no" ]
)
dnl Multi-GPU options
dnl ==================

dnl enable multi-gpu mode
AC_ARG_ENABLE(multi-gpu, 
  AC_HELP_STRING([--enable-multi-gpu], [ Enable Multi-GPU mode (default is no)]),
  [ multi_gpu=${enableval} ],
  [ multi_gpu="no" ]
)

dnl enable comms compute/overlap
AC_ARG_ENABLE(overlap-comms,
  AC_HELP_STRING([--enable-overlap-comms], [ Enable comms/compute overlap (default is yes)]),
  [ overlap_comms=${enableval}],
  [ overlap_comms="yes" ]
)

AC_ARG_WITH(mpi,
 AC_HELP_STRING([--with-mpi= MPIDIR], [ Specify installation directory for MPI ]),
 [ mpi_home=${withval}; build_mpi="yes" ],
 [ mpi_home=""; build_mpi="no" ]
) 

AC_ARG_WITH(qmp,
 AC_HELP_STRING([--with-qmp=QMPDIR], [ Specify QMP installation directory ]),
 [ qmp_home=${withval} ; build_qmp="yes" ],
 [ qmp_home="" ; build_qmp="no" ]
)

AC_ARG_WITH(qio,
 AC_HELP_STRING([--with-qio=QIODIR], [ Specify QIO installation directory ]),
 [ qio_home=${withval} ; build_qio="yes" ],
 [ qio_home="" ; build_qio="no" ]
)

AC_ARG_ENABLE(fermi-double-tex,
 AC_HELP_STRING([--enable-fermi-double-tex], [ Enable double-precision texture reads on Fermi (default is no) ]),
 [ fermi_double_tex=${enableval}],
 [ fermi_double_tex="no" ]
)


dnl Input validation

dnl CPU Arch
case ${cpu_arch} in
x86 | x86_64 ) ;;
*) 
  AC_MSG_ERROR([ cpu arch must be either x86_64 or x86 ])
  ;;
esac

dnl CPU Arch
case ${gpu_arch} in
sm_10 | sm_11 | sm_12 | sm_13 | sm_20 | sm_21 ) ;;
*) 
  AC_MSG_ERROR([ cpu arch must be either x86_64 or x86 ])
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
  
  AC_MSG_NOTICE([Multi-GPU doesnt support Domain-Wall Dirac Op  yet: disabling ])
  build_dwf_dirac="no";

  AC_MSG_NOTICE([Multi-GPU doesnt support Gauge Force yet: disabling ])
  build_gauge_force="no"

  AC_MSG_NOTICE([Multi-GPU doesnt support Staggered Force yet:disabling ])
  build_staggered_force="no"


  if test "X${qmp_home}X" = "XX"; then
    if test "X${mpi_home}X" = "XX"; then
        AC_MSG_WARN([ Multi-GPU build without QMP or MPI. Will build single node code with copies ])
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
 
dnl Enables overlapping comms
case ${fermi_double_tex} in
yes|no);;
*) 
  AC_MSG_ERROR([ invalid value for --enable-fermi-double-tex ])
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

AC_MSG_NOTICE([Setting DEVICE = ${tune_device} ])
AC_SUBST( DEVICE,[${tune_device}] )

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

AC_MSG_NOTICE([Setting BUILD_QIO = ${build_qio} ])
AC_SUBST( BUILD_QIO, [${build_qio}])

AC_MSG_NOTICE([Setting FECC = ${CC} ])
AC_SUBST( FECC, [${CC}])

AC_MSG_NOTICE([Setting FECXX = ${CXX} ])
AC_SUBST( FECXX, [${CXX}])

AC_MSG_NOTICE([Setting MPI_HOME=${mpi_home}])
AC_SUBST( MPI_HOME, [${mpi_home}] )

AC_MSG_NOTICE([Setting QMP_HOME=${qmp_home}])
AC_SUBST( QMP_HOME, [${qmp_home}] )

AC_MSG_NOTICE([Setting QIO_HOME=${qio_home}])
AC_SUBST( QIO_HOME, [${qio_home}] )

AC_MSG_NOTICE([Setting HAVE_NUMA=${ac_cv_header_numa_h}])
AC_SUBST( HAVE_NUMA, [${ac_cv_header_numa_h}] )

AC_MSG_NOTICE([Setting FERMI_DBLE_TEX= ${fermi_double_tex}])
AC_SUBST( FERMI_DBLE_TEX, [${fermi_double_tex}])

AC_CONFIG_FILES(make.inc)
AC_OUTPUT
