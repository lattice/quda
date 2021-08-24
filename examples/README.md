# QUDA examples 

## Overview

The routines in this directory are examples of how to write your own C++ code that
utilise the QUDA library. They rely heavily on the host routines in QUDA's test
directory, but that is just for convenience: it is possible to write code that is
independent of those utilites and links to libquda. Included in this directory are:

* propagator_quda.cpp
* correlator_quda.cpp

which are carbon copies of the corresponding test routines, and

* heatbath_quda.cpp
* wilsonflow_quda.cpp

which are trimmed down versions of code in the tests. These files demonstrate not
only how to invoke particular computations in QUDA, but also how to link C++ code
to QUDA and how to construct QUDA data objects such as gauge fields and vectors.
It is hoped that these examples will serve both as stand-alone routines for
common LQCD computations, and as platfroms from which users may construct more
elaborate or specific workflows.

## Building

The sole dependency is QUDA, which should be built according to the following
instructions
```
https://github.com/lattice/quda/wiki/Building-QUDA-with-cmake
```

QUDA has automated scripts to download and compile its own dependencies
https://github.com/lattice/quda/wiki/QUDA-%28Optional%29-Dependencies
We give the option to link against the QMP and QIO dependencies.
Once QUDA is build and/or installed, you create a build directory for the examples
in the location of your choosing, then invoke the CMakeLists.txt file in `quda/examples`
```
mkdir build
cd build
ccmake </path/to/quda/examples>
```
From here you need to set `QUDA_BUILD_HOME` and `QUDA_SOURCE_HOME` to be the installation
or build directory of QUDA, and the QUDA source respectively. Other relevant options are
```
USE_QMP
USE_MPI
USE_QIO
```
which require the corresponding `QUDA_QMP`, `QUDA_MPI`, and `QUDA_QIO` options to be
enabled in the QUDA build. We give here an invocation that will build the examples
with `QIO` and `QMP` enabled:
```
cmake /path/to/quda/examples -DQUDA_SOURCE_HOME=/path/to/quda -DQUDA_BUILD_HOME=path/to/quda_build -DUSE_QMP=ON -DUSE_QIO=ON
```

## Running

The example routines are designed to behave exactly like QUDA test routines. One can
invoke a help message by running an executable with the `--help` flag:
```
./<example>_quda --help
```
which will give an exhaustive list of options. 
