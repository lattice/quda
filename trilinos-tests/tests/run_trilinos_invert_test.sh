
export QUDA_RESOURCE_PATH=./
export QUDA_ENABLE_TUNING=1

#WARNING use case 3 for MdagM application
#1
#./trilinos_invert_test --prec double --dslash-type wilson --dim 16 16 16 16 --test 3 --mass -0.1
#2
#./trilinos_invert_test --prec double --dslash-type wilson --dim 16 16 16 16 --test 3 --mass -0.8
#3
./trilinos_invert_test --prec double --dslash-type wilson --dim 16 16 16 8 --test 3 --mass -0.8

