#!/bin/bash

OUTFILE=/tmp/quda_test.txt
echo "" > $OUTFILE

fail_msg="*****************************Sanity check for quda failed!*****************************"
success_msg="Sanity check for quda passed!"

progs="staggered_dslash_test staggered_invert_test llfat_test gauge_force_test fermion_force_test"
precs="double"
recons="18"

for prog in $progs ; do 
  if [ ! -e "$prog" ]; then
	echo "The program $prog not found!"
	continue
  fi

  if [ "$prog" = "llfat_test" -o "$prog" = "gauge_force_test" -o "$prog" = "fermion_force_test" ]; then
	extra_args="--verify"
  else
	extra_args=""
  fi 

  for prec in $precs ; do 
    for recon in $recons ; do

	#FIXME: now we only support 12-recon and single precision
        if [ "$prog" = "gauge_force_test" -o "$prog" = "fermion_force_test" ]; then
	  prec="single"	
	  recon="12"
        fi


	cmd="./$prog --sdim 8 --tdim 16 --recon $recon --prec $prec $extra_args"
	echo -ne  $cmd  "\t"..."\t"
	echo "----------------------------------------------------------" >>$OUTFILE
        echo $cmd >> $OUTFILE
	$cmd >> $OUTFILE 2>&1|| (echo -e "FAIL\n$prog failed, check $OUTFILE for detail"; echo $fail_msg; exit 1) || exit 1
        echo "OK"
    done
  done
done

echo $success_msg
