# QUDA 1.1.0

## Overview

QUDA is a library for performing calculations in lattice QCD on graphics
processing units (GPUs), leveraging NVIDIA's CUDA platform. The current
release includes optimized Dirac operators and solvers for the following
fermion actions:

* Wilson 
* Clover-improved Wilson
* Twisted mass (including non-degenerate pairs)
* Twisted mass with a clover term 
* Staggered fermions
* Improved staggered (asqtad or HISQ) 
* Domain wall (4-d or 5-d preconditioned)
* Möbius fermion

Implementations of CG, multi-shift CG, BiCGStab, BiCGStab(l), and
DD-preconditioned GCR are provided, including robust mixed-precision
variants supporting combinations of double, single, half and quarter
precisions (where the latter two are 16-bit and 8-bit "block floating
point", respectively).  The library also includes auxiliary routines
necessary for Hybrid Monte Carlo, such as HISQ link fattening, force
terms and clover- field construction.  Use of many GPUs in parallel is
supported throughout, with communication handled by QMP or MPI.

QUDA includes an implementations of adaptive multigrid for the Wilson,
clover-improved, twisted-mass and twisted-clover fermion actions.  We
note however that this is undergoing continued evolution and
improvement and we highly recommend using adaptive multigrid use the
latest develop branch.  More details can be found [here]
(https://github.com/lattice/quda/wiki/Multigrid-Solver).

Support for eigen-vector deflation solvers is also included through
the Thick Restarted Lanczos Method (TRLM), and we offer an Implicitly
Restarted Arnoldi for observing non-hermitian operator spectra.
For more details we refer the user to the wiki:
[QUDA's eigensolvers]
(https://github.com/lattice/quda/wiki/QUDA%27s-eigensolvers)
[Deflating coarse grid solves in Multigrid]
(https://github.com/lattice/quda/wiki/Multigrid-Solver#multigrid-inverter--lanczos)

## Software Compatibility:

The library has been tested under Linux (CentOS 7 and Ubuntu 18.04)
using releases 10.1 through 11.4 of the CUDA toolkit.  Earlier versions
of the CUDA toolkit will not work, and we highly recommend the use of
11.x.  QUDA has been tested in conjunction with x86-64, IBM
POWER8/POWER9 and ARM CPUs.  Both GCC and Clang host compilers are
supported, with the mininum recommended versions being 7.x and 6, respectively.
CMake 3.15 or greater to required to build QUDA.

See also Known Issues below.


## Hardware Compatibility:

For a list of supported devices, see

http://developer.nvidia.com/cuda-gpus

Before building the library, you should determine the "compute
capability" of your card, either from NVIDIA's documentation or by
running the deviceQuery example in the CUDA SDK, and pass the
appropriate value to the `QUDA_GPU_ARCH` variable in cmake.

QUDA 1.1.0, supports devices of compute capability 3.0 or greater.
QUDA is no longer supported on the older Tesla (1.x) and Fermi (2.x)
architectures.

See also "Known Issues" below.


## Installation:

It is recommended to build QUDA in a separate directory from the
source directory.  For instructions on how to build QUDA using cmake
see this page
https://github.com/lattice/quda/wiki/QUDA-Build-With-CMake. Note
that this requires cmake version 3.15 or later. You can obtain cmake
from https://cmake.org/download/. On Linux the binary tar.gz archives
unpack into a cmake directory and usually run fine from that
directory.

The basic steps for building with cmake are:

1. Create a build dir, outside of the quda source directory. 
2. In your build-dir run `cmake <path-to-quda-src>` 
3. It is recommended to set options by calling `ccmake` in
your build dir. Alternatively you can use the `-DVARIABLE=value`
syntax in the previous step.
4. run 'make -j <N>' to build with N
parallel jobs. 
5. Now is a good time to get a coffee.

You are most likely to want to specify the GPU architecture of the
machine you are building for. Either configure QUDA_GPU_ARCH in step 3
or specify e.g. -DQUDA_GPU_ARCH=sm_60 for a Pascal GPU in step 2.

### Multi-GPU support

QUDA supports using multiple GPUs through MPI and QMP, together with
the optional use of NVSHMEM GPU-initiated communication for improved
strong scaling of the Dirac operators.  To enable multi-GPU support
either set `QUDA_MPI` or `QUDA_QMP` to ON when configuring QUDA
through cmake.

Note that in any case cmake will automatically try to detect your MPI
installation. If you need to specify a particular MPI please set
`MPI_C_COMPILER` and `MPI_CXX_COMPILER` in cmake.  See also
https://cmake.org/cmake/help/v3.9/module/FindMPI.html for more help.

For QMP please set `QUDA_QMP_HOME` to the installation directory of QMP.

For more details see https://github.com/lattice/quda/wiki/Multi-GPU-Support

To enable NVSHMEM support set `QUDA_NVSHMEM` to ON, and set the
location of the local NVSHMEM installation with `QUDA_NVSHMEM_HOME`.
For more details see
https://github.com/lattice/quda/wiki/Multi-GPU-with-NVSHMEM

### External dependencies

The eigen-vector solvers (eigCG and incremental eigCG) by default will
use Eigen, however, QUDA can be configured to use MAGMA if available
(see https://github.com/lattice/quda/wiki/Deflated-Solvers for more
details).  MAGMA is available from
http://icl.cs.utk.edu/magma/index.html.  MAGMA is enabled using the
cmake option `QUDA_MAGMA=ON`.

Version 1.1.0 of QUDA includes interface for the external (P)ARPACK
library for eigenvector computing. (P)ARPACK is available, e.g., from
https://github.com/opencollab/arpack-ng.  (P)ARPACK is enabled using
CMake option `QUDA_ARPACK=ON`. Note that with a multi-GPU option, the
build system will automatically use PARPACK library.

Automatic download and installation of Eigen, (P)ARPACK, QMP and QIO
is supported in QUDA through the CMake options QUDA_DOWNLOAD_EIGEN,
QUDA_DOWNLOAD_ARPACK, and QUDA_DOWNLOAD_USQCD.

### Application Interfaces

By default only the QDP and MILC interfaces are enabled.  For
interfacing support with QDPJIT, BQCD, CPS or TIFR; this should be
enabled at by setting the corresponding `QUDA_INTERFACE_<application>`
variable e.g., `QUDA_INTERFACE_BQCD=ON`.  To keep compilation time to
a minimum it is recommended to only enable those interfaces that are
used by a given application.

## Tuning

Throughout the library, auto-tuning is used to select optimal launch
parameters for most performance-critical kernels.  This tuning process
takes some time and will generally slow things down the first time a
given kernel is called during a run.  To avoid this one-time overhead in
subsequent runs (using the same action, solver, lattice volume, etc.),
the optimal parameters are cached to disk.  For this to work, the
`QUDA_RESOURCE_PATH` environment variable must be set, pointing to a
writable directory.  Note that since the tuned parameters are hardware-
specific, this "resource directory" should not be shared between jobs
running on different systems (e.g., two clusters with different GPUs
installed).  Attempting to use parameters tuned for one card on a
different card may lead to unexpected errors.

This autotuning information can also be used to build up a first-order
kernel profile: since the autotuner measures how long a kernel takes
to run, if we simply keep track of the number of kernel calls, from
the product of these two quantities we have a time profile of a given
job run.  If `QUDA_RESOURCE_PATH` is set, then this profiling
information is output to the file "profile.tsv" in this specified
directory.  Optionally, the output filename can be specified using the
`QUDA_PROFILE_OUTPUT` environment variable, to avoid overwriting
previously generated profile outputs.  In addition to the kernel
profile, a policy profile, e.g., collections of kernels and/or other
algorithms that are auto-tuned, is also output to the file
"profile_async.tsv".  The policy profile for example includes
the entire multi-GPU dslash, whose style and order of communication is
autotuned.  Hence while the dslash kernel entries appearing the kernel
profile do include communication time, the entries in the policy
profile include all constituent parts (halo packing, interior update,
communication and exterior update).

## Using the Library:

Include the header file include/quda.h in your application, link against
lib/libquda.so, and study tests/invert_test.cpp (for Wilson, clover,
twisted-mass, or domain wall fermions) or
tests/staggered_invert_test.cpp (for asqtad/HISQ fermions) for examples
of the solver interface.  The various solver options are enumerated in
include/enum_quda.h.


## Known Issues:

* When the auto-tuner is active in a multi-GPU run it may cause issues
with binary reproducibility of this run if domain-decomposition
preconditioning is used. This is caused by the possibility of
different launch configurations being used on different GPUs in the
tuning run simultaneously. If binary reproducibility is strictly
required make sure that a run with active tuning has completed. This
will ensure that the same launch configurations for a given kernel is
used on all GPUs and binary reproducibility.

## Getting Help:

Please visit http://lattice.github.io/quda for contact information. Bug
reports are especially welcome.


## Acknowledging QUDA:

[![DOI](https://zenodo.org/badge/1300564.svg)](https://zenodo.org/badge/latestdoi/1300564)
  
If you find this software useful in your work, please cite:

M. A. Clark, R. Babich, K. Barros, R. Brower, and C. Rebbi, "Solving
Lattice QCD systems of equations using mixed precision solvers on GPUs,"
Comput. Phys. Commun. 181, 1517 (2010) [arXiv:0911.3191 [hep-lat]].

When taking advantage of multi-GPU support, please also cite:

R. Babich, M. A. Clark, B. Joo, G. Shi, R. C. Brower, and S. Gottlieb,
"Scaling lattice QCD beyond 100 GPUs," International Conference for High
Performance Computing, Networking, Storage and Analysis (SC), 2011
[arXiv:1109.2935 [hep-lat]].

When taking advantage of adaptive multigrid, please also cite:

M. A. Clark, B. Joo, A. Strelchenko, M. Cheng, A. Gambhir, and R. Brower,
"Accelerating Lattice QCD Multigrid on GPUs Using Fine-Grained
Parallelization," International Conference for High Performance
Computing, Networking, Storage and Analysis (SC), 2016
[arXiv:1612.07873 [hep-lat]].

When taking advantage of block CG, please also cite:

M. A. Clark, A. Strelchenko, A. Vaquero, M. Wagner, and E. Weinberg,
"Pushing Memory Bandwidth Limitations Through Efficient
Implementations of Block-Krylov Space Solvers on GPUs,"
Comput. Phys. Commun. 233 (2018), 29-40 [arXiv:1710.09745 [hep-lat]].

When taking advantage of the Möbius MSPCG solver, please also cite:

Jiqun Tu, M. A. Clark, Chulwoo Jung, Robert Mawhinney, "Solving DWF
Dirac Equation Using Multi-splitting Preconditioned Conjugate Gradient
with Tensor Cores on NVIDIA GPUs," published in the Platform of
Advanced Scientific Computing (PASC21) [arXiv:2104.05615[hep-lat]].


## Authors:

*  Ronald Babich (NVIDIA)
*  Simone Bacchio (Cyprus)
*  Michael Baldhauf (Regensburg)
*  Kipton Barros (Los Alamos National Laboratory)
*  Richard Brower (Boston University) 
*  Nuno Cardoso (NCSA) 
*  Kate Clark (NVIDIA)
*  Michael Cheng (Boston University)
*  Carleton DeTar (Utah University)
*  Justin Foley (NIH)
*  Arjun Gambhir (William and Mary)
*  Marco Garofalo (HISKP, University of Bonn)
*  Joel Giedt (Rensselaer Polytechnic Institute) 
*  Steven Gottlieb (Indiana University) 
*  Kyriakos Hadjiyiannakou (Cyprus)
*  Dean Howarth (Lawrence Livermore Lab, Lawrence Berkeley Lab)
*  Hwancheol Jeong (Indiana University)
*  Xiangyu Jiang (Chinese Academy of Sciences)
*  Balint Joo (OLCF, Oak Ridge National Laboratory, formerly Jefferson Lab)
*  Hyung-Jin Kim (Samsung Advanced Institute of Technology)
*  Bartosz Kostrzewa (HPC/A-Lab, University of Bonn)
*  James Osborn (Argonne National Laboratory)
*  Ferenc Pittler (Cyprus)
*  Claudio Rebbi (Boston University) 
*  Eloy Romero (William and Mary)
*  Hauke Sandmeyer (Bielefeld)
*  Mario Schröck (INFN)
*  Guochun Shi (NCSA)
*  Alexei Strelchenko (Fermi National Accelerator Laboratory)
*  Jiqun Tu (NVIDIA)
*  Carsten Urbach (HISKP, University of Bonn)
*  Alejandro Vaquero (Utah University)
*  Mathias Wagner (NVIDIA)
*  Andre Walker-Loud (Lawrence Berkley Laboratory)
*  Evan Weinberg (NVIDIA)
*  Frank Winter (Jefferson Lab)
*  Yi-Bo Yang (Chinese Academy of Sciences)
*  Anthony Grebe (Fermilab)
*  Michael Wagman (Fermilab)


Portions of this software were developed at the Innovative Systems Lab,
National Center for Supercomputing Applications
http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html

Development was supported in part by the U.S. Department of Energy under
grants DE-FC02-06ER41440, DE-FC02-06ER41449, and DE-AC05-06OR23177; the
National Science Foundation under grants DGE-0221680, PHY-0427646,
PHY-0835713, OCI-0946441, and OCI-1060067; as well as the PRACE project
funded in part by the EUs 7th Framework Programme (FP7/2007-2013) under
grants RI-211528 and FP7-261557.  Any opinions, findings, and
conclusions or recommendations expressed in this material are those of
the authors and do not necessarily reflect the views of the Department
of Energy, the National Science Foundation, or the PRACE project.
