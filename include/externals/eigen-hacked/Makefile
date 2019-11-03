# Thea -*- Makefile -*-
#
# By Shun Xu, 9 Mar 2019

SSH_CMD=ssh -q
SSH_HOST=sccas@r4-node9
SSH_DIR=/home/sccas/alwin/xushun/eigen_rocm 

SRC_CODE = $(shell git ls-files .|grep -v -E 'COPYING|Makefile|CMakeLists.txt|INSTALL|README.md' )
SUB_HIP=$(shell if (which hipcc >/dev/null 2>&1) ; then echo "_hip" ; fi )

PACKAGE_VERSION=3.3.90
LQCD_INSTALL=/home/sccas/alwin/latticeQCD/bin
EIGEN_INCLUDE_DIR=/home/sccas/alwin/xushun/eigen_rocm
PWD_DIR=$(shell pwd)

all:
	rm -rf build && mkdir -p build/tmp
##
clean:
	rm -rf build*
## 
tar:
	git archive HEAD --format=tar.gz --prefix=hup-$(PACKAGE_VERSION)/ --verbose -o eigen-$(PACKAGE_VERSION).tar.gz
lines:
	@cat $(SRC_CODE) |wc -l
	@echo "without blank lines"
	@cat $(SRC_CODE) |grep -v ^$$|wc -l
files:
	git ls-files |wc -l
##
rsync:
	git ls-files -coz --directory --exclude-standard | rsync --verbose -ar0 --files-from=- -e '${SSH_CMD}' ./ ${SSH_HOST}:${SSH_DIR}
rbuild:rsync
	${SSH_CMD} ${SSH_HOST} make -C $(SSH_DIR) all
