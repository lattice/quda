#!/bin/bash

CUDA_HOME=/usr/local/cuda-10.1
QUDA_HOME=/home/astrel/Workspace/QUDA-oneapi/quda-oneapi-upd-3
PROJ_DIR=$QUDA_HOME/lib

test_version_=$(date +%s)

OUTDIR=dpcpp_output_${test_version_}

dpct -p -in-root=$PROJ_DIR -out-root=$OUTDIR \
     --keep-original-code \
     --extra-arg='-I$QUDA_HOME/include' \
     --extra-arg='-I./include' \
     --extra-arg='-I./include/externals' \
     --extra-arg='-I./lib' \
     --extra-arg='-std=c++14' \
     $QUDA_HOME/lib/color_spinor_util.cu


rm dpcpp_out
ln -sf ./$OUTDIR dpcpp_out
