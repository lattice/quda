#!/bin/bash

OUTFILE=/tmp/quda_test.txt
echo "" > $OUTFILE

fail_msg="*****************************Sanity check for quda failed! *****************************"
success_msg="Sanity check for quda passed!"

progs="staggered_dslash_test staggered_invert_test"
precs="double"
recons="18"

echo "Testing staggered dslash and inverter:"
for prog in $progs ; do 
  if [ ! -e "$prog" ]; then
	echo "The program $prog does not exist; this program will not be tested!"
	continue
  fi
  for prec in $precs ; do 
    for recon in $recons ; do
	cmd="mpirun -n 1 ./$prog --sdim 8 --tdim 16 --recon $recon --prec $prec"
	echo running $cmd
	echo "----------------------------------------------------------" >>$OUTFILE
        echo $cmd >> $OUTFILE
	$cmd >> $OUTFILE 2>&1|| (echo "$prog failed, check $OUTFILE for detail"; echo $fail_msg; exit 1) || exit 1
    done
  done
done


echo "Testing link fattening code"
precs="double"
recons="18"

file=./llfat_test
if [  -e "$file" ]; then
  for prec in $precs ; do
    for recon in $recons ; do
      cmd="$file --sdim 8 --tdim 16 --recon $recon --cpu_prec ${prec} --prec ${prec} --verify"
      echo running $cmd
      echo $cmd >>$OUTFILE
      $cmd >>  $OUTFILE 2>&1|| (echo "$prog failed, check $OUTFILE for detail"; echo $fail_msg; exit 1) || exit 1
    done
  done
else
  echo "The program $file does not exist; this program will not be tested!"
fi

echo $success_msg
