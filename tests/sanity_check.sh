#!/bin/bash

OUTFILE=/tmp/quda_test.txt
echo "" > $OUTFILE

fail_msg="*****************************Sanity check for quda failed!*****************************"

function basic_sanity_check {

    echo "Performing basic sanity test:"
    progs="staggered_dslash_test staggered_invert_test llfat_test llfat_test gauge_force_test fermion_force_test"
    precs="double"
    recons="18"
    whichtest="1"
	
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

	if [ "$prog" = "llfat_test" ]; then
	    whichtest=$((1-$whichtest))	
	    extra_args+="  --test $whichtest"
	fi

        for prec in $precs ; do 
            for recon in $recons ; do

	        #FIXME: now we only support 12-recon and single precision
                if [ "$prog" = "fermion_force_test" ]; then
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
  echo "output file: $OUTFILE"
  echo "Basic sanity check for quda passed!"
}

function complete_fatlink_check {
    echo "Performing complete fatlink test:"	
    prog="./llfat_test"
    precs="double single"
    recons="18 12"
    tests="0 1" 
    for prec in $precs; do 
	for recon in $recons; do 
	    for tst in $tests; do
		cmd="$prog --sdim 8 --tdim 16 --prec $prec --recon $recon --test $tst --verify"
		echo -ne  $cmd  "\t"..."\t"
		echo "----------------------------------------------------------" >>$OUTFILE
		echo $cmd >> $OUTFILE
		$cmd >> $OUTFILE 2>&1|| (echo -e "FAIL\n$prog failed, check $OUTFILE for detail"; echo $fail_msg; exit 1) || exit 1
		echo "OK"
	    done
	done
    done

}

function complete_dslash_check {
    echo "Performing complete dslash test:"
    prog="./staggered_dslash_test"
    precs="double single half"
    recons="18 12"
    tests="0 1"
    for prec in $precs; do
        for recon in $recons; do
            for tst in $tests; do
                cmd="$prog --sdim 8 --tdim 16 --prec $prec --recon $recon --test $tst"
                echo -ne  $cmd  "\t"..."\t"
                echo "----------------------------------------------------------" >>$OUTFILE
                echo $cmd >> $OUTFILE
                $cmd >> $OUTFILE 2>&1|| (echo -e "FAIL\n$prog failed, check $OUTFILE for detail"; echo $fail_msg; exit 1) || exit 1
                echo "OK"
            done
        done
    done

}

function complete_invert_check {
    echo "Performing complete invert test:"
    prog="./staggered_invert_test"
    precs="double single half"
    recons="18 12"
    tests="0 1 3 4 6"
    partitions="0 8"
    for prec in $precs; do
        for recon in $recons; do
            for tst in $tests; do
		for partition in $partitions; do 
		    tol="1e-6"
                    if [ $prec == "half" ] ; then
			tol="1e-3"
                    fi
                    cmd="$prog --sdim 8 --tdim 16 --prec $prec --recon $recon --test $tst --tol $tol --partition $partition"
                    echo -ne  $cmd  "\t"..."\t"
                    echo "----------------------------------------------------------" >>$OUTFILE
                    echo $cmd >> $OUTFILE
                    $cmd >> $OUTFILE 2>&1|| (echo -e "FAIL\n$prog failed, check $OUTFILE for detail"; echo $fail_msg; exit 1) || exit 1
                    echo "OK"
		done
            done
        done
    done

   

}
#actions based on arguments

if [ $# == "0" ]; then
	basic_sanity_check
	exit
fi 

action=$1
case $action in
    basic )
	basic_sanity_check ;;
    fat )
	complete_fatlink_check ;;
    dslash )
        complete_dslash_check ;;
    invert )
	complete_invert_check;;
    all )
	basic_sanity_check
	complete_fatlink_check
	complete_dslash_check
	complete_invert_check
	;;
esac




