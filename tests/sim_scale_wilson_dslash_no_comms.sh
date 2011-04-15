#!/bin/bash
nx=32
ny=32
nz=32
nt=256

function set_partition_mask {
    gs_x=$1
    gs_y=$2
    gs_z=$3
    gs_t=$4

    mask=0

    if [ $gs_x -ne 1 ]; then
	mask=$(($mask+1))
    fi

    if [ $gs_y -ne 1 ]; then
	mask=$(($mask+2))
    fi

    if [ $gs_z -ne 1 ]; then
	mask=$(($mask+4))
    fi

    if [ $gs_t -ne 1 ]; then
	mask=$(($mask+8))
    fi
}
    
function run_dslash_test {
    #ngpus=$1
    ngpus=1
    gs_x=$2
    gs_y=$3
    gs_z=$4
    gs_t=$5

    xdim=$(($nx/$gs_x))
    ydim=$(($ny/$gs_y))
    zdim=$(($nz/$gs_z))
    tdim=$(($nt/$gs_t))

#    set_partition_mask $gs_x $gs_y $gs_z $gs_t
    mask=0
    
    precs="double single half"
    recons="18 12"
    
    prog="wilson_dslash_test"
    if [ ! -e "$prog" ]; then
        echo "The program $prog does not exist; this program will not be tested!"
        exit
    fi
    for prec in $precs ; do
	for recon in $recons ; do
	    cmd="mpirun -n $ngpus ./$prog --dslash_type clover --xdim $xdim --ydim $ydim --zdim $zdim --tdim $tdim --recon $recon --prec $prec --partition $mask"
#	    echo running $cmd
	    echo "----------------------------------------------------------" 
	    echo $cmd 
	    $cmd 
	done
    done
}

#gridsize for 8 GPUs
partitions[8]="
1 1 2 4 
1 1 1 8 
"

#gridsize for 16 GPUs
partitions[16]="
1 1 2 8 
1 1 1 16
"

#gridsize for 32 GPUs
partitions[32]="
1 1 2 16 
1 1 1 32 
"

#gridsize for 64 GPUs
partitions[64]="
1 2 2 16
1 1 2 32
1 1 1 64
"
#gridsize for 128 GPUs
partitions[128]="
2 2 2 16 
1 2 2 32
1 1 4 32
1 1 1 128
"
#gridsize for 256  GPUs
partitions[256]="
2 2 2 32
1 2 4 32 
1 1 4 64
"


num_gpus="256 128 64 32 16 8"
for n in $num_gpus; do 
    #echo for $n GPUs
    plist=${partitions[$n]}
    j="0"	
    for v in $plist; do
	modj=$(($j % 4))
	if [ $modj == "0" ]; then
	    gridsize_x=$v
	fi
	
        if [ $modj == "1" ]; then
	    gridsize_y=$v
        fi
	
        if [ $modj == "2" ]; then
	    gridsize_z=$v
        fi
	
        if [ $modj == "3" ]; then
	    gridsize_t=$v
	    #echo gridsize_x/y/z/t/=$gridsize_x, $gridsize_y, $gridsize_z, $gridsize_t
	    run_dslash_test $n $gridsize_x $gridsize_y $gridsize_z $gridsize_t
        fi
	
	j=$(($j+1))
  done
done


