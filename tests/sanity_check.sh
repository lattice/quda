#!/bin/bash

OUTFILE=/tmp/quda_test.txt
echo "" > $OUTFILE

fail_msg="Sanity check for quda failed!"
success_msg="Sanity check for quda passed!"

progs="staggered_dslash_test staggered_invert_test"
precs="double single"
recons="18 12 8"

echo "Testing staggered dslash and inverter:"
for prog in $progs ; do 
  for prec in $precs ; do 
    for recon in $recons ; do
	cmd="$prog --sdim 8 --tdim 16 --recon $recon --prec $prec"
	echo running $cmd
	./$cmd >> $OUTFILE 2>&1|| (echo "$prog failed, check $OUTFILE for detail"; echo $fail_msg; exit 1) || exit 1
    done
  done
done

echo $success_msg
