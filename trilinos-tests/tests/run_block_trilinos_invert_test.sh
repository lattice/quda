
export QUDA_RESOURCE_PATH=./
export QUDA_ENABLE_TUNING=1

#WARNING use case 3 for MdagM application
Nsrc=2
Nblk=2
./trilinos_block_invert_test --prec double --dslash-type wilson --dim 16 16 16 16 --test 3 --mass -0.8 --tol 1e-10 --niter 600 --nsrc ${Nsrc} --msrc ${Nblk}

