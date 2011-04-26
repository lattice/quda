#include <unistd.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <numa.h>
#include <stdio.h>
#include <sys/time.h>
#include <errno.h>

#define CUERR  do{ cudaError_t err;					\
        if ((err = cudaGetLastError()) != cudaSuccess) {		\
	    printf("ERROR: CUDA error: %s, line %d\n", cudaGetErrorString(err), __LINE__); \
	    exit(-1);							\
        }}while(0)

#define MAX_NODES   32
#define MAX_STRING_LEN 128
#define VERBOSE_PRINT(fmt,...) do {		\
	if (verbose >=2){			\
	    printf(fmt, ##__VA_ARGS__);		\
	}					\
    }while(0)

#define PRINT(fmt,...) do {			\
	if (verbose){				\
	    printf(fmt, ##__VA_ARGS__);		\
	}					\
    }while(0)

static int verbose = 0;

static double
gettime(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);

    return t.tv_sec + 0.000001* t.tv_usec;
}


/* This function tests the gpu bandwidth (H2D and D2H) with given cpu bitmask
 * The highest number on either direction are summed together
 */

double
bandwidth_test(int node, unsigned long * bitmask, int deviceid)
{
    double sum;
    char* A = NULL;
    char* hostA = NULL;
    char* hostB = NULL;
    double t0, t1;
    unsigned int max_data_size = 1 << 28;

    int i;
    int repeat_num = 100;
    int cpu_index;
    
/*
#if (LIBNUMA_API_VERSION >=2)
    numa_sched_setaffinity(0, (struct bitmask*)bitmask);
#else
    //no corresponding function here
#endif
*/

    numa_run_on_node(node);
    
    unsigned int datasize = 1048576;
    cudaSetDevice(deviceid);CUERR;
    
    cudaMallocHost((void**)&hostA, max_data_size); CUERR;
    cudaMallocHost((void**)&hostB, max_data_size); CUERR;
    cudaMalloc((void**)&A, max_data_size); CUERR;
    
    VERBOSE_PRINT("\nTesting node=%d, gpu deviceid =%d\n",node, deviceid);
    VERBOSE_PRINT("size(bytes) \th2d_time(us) \t bandwidth(MB/s)\td2htime(us) \t banwidth(MB/s)\n");
    
    
    double max_h2d_bandwidth = 0;
    double max_d2h_bandwidth = 0;

    while( datasize <= max_data_size){
	repeat_num = 10;
	
	t0 = gettime();
	for (i =0 ;i <repeat_num ;i ++){
	    cudaMemcpy(A, hostA, datasize, cudaMemcpyHostToDevice);
	    cudaThreadSynchronize();
	}
	t1 = gettime();
	
	double h2d_time = t1 - t0;
	
	t0 = gettime();
	for (i =0 ;i <repeat_num ;i ++){
	    cudaMemcpy(hostB, A, datasize, cudaMemcpyDeviceToHost);
	    cudaThreadSynchronize();
	}
	t1 = gettime();
	
	double d2h_time = t1 - t0;	
	
	double h2d_bandwidth = 1.0*datasize*repeat_num/h2d_time*1e-6;
	double d2h_bandwidth = 1.0*datasize*repeat_num/d2h_time*1e-6;
	
	if (h2d_bandwidth> max_h2d_bandwidth){
	    max_h2d_bandwidth = h2d_bandwidth;
	}
	
	if (d2h_bandwidth > max_d2h_bandwidth){
	    max_d2h_bandwidth = d2h_bandwidth;
	}
	
	VERBOSE_PRINT("%10d \t%10.2f \t%10.2f \t%10.2f \t%10.2f\n",
	       datasize,
	       h2d_time/repeat_num*(1e+6), h2d_bandwidth,
	       d2h_time/repeat_num*(1e+6), d2h_bandwidth );
	fflush(stdout);
	
	datasize = datasize* 2;
    }
       
    sum = max_h2d_bandwidth + max_d2h_bandwidth;
    
    
    cudaFreeHost(hostA);
    cudaFreeHost(hostB);
    cudaFree(A);    
    
    cudaThreadExit();
    
    VERBOSE_PRINT(" max_d2h_bandwidth=%.f, max_h2d_bandwidht=%.f, sum=%.f\n",
	   max_d2h_bandwidth, max_h2d_bandwidth, sum);

    return sum;

}

void
process_bandwidth_table(double * bandwidth_table, int nodes, int device_count, 
			int* gpu_affinity_table)
{
    int i, j;
    for(i=0;i < device_count; i++){
	double max = 0;
	int max_idx = 0;
	for(j=0; j < nodes; j++){
	    if (bandwidth_table[i*nodes + j]> max){
		max = bandwidth_table[i*nodes + j];
		max_idx = j;
	    }   
	}	
	gpu_affinity_table[i]= max_idx;
    }
    
    return;
}

void 
display_bandwidth_table(double* table, int nodes, int device_count)
{
    int i, j;
    PRINT("Displaying bandwidth table: \n");
    for(i=0; i < nodes; i++){
	PRINT("\tnode%d", i);
    }
    PRINT("\n");
    
    for(i =0;i < device_count; i++){
	PRINT("gpu%d\t", i);
	for(j=0; j < nodes; j++){
	    PRINT("%.f\t", table[i*nodes + j]);   
	}
	PRINT("\n");
    }
 

   PRINT("\n");   
}

void
output_to_file(char node_affinity_string[MAX_NODES][MAX_STRING_LEN], int* gpu_affinity_table, int device_count, char* output_file)
{
    char filebuf[1024*16];       
    char* s =  filebuf;
    char buf[1024];
    int buflen=1024;
    int i;
    FILE* file = fopen(output_file, "r");
    if (file){
	while(fgets(buf, buflen, file) != NULL){
	    char keyword[] = "affinity";
	    if (strncmp(buf, keyword, strlen(keyword)) !=0){
		s+= sprintf(s,"%s", buf);
	    }
	}
	fclose(file);
    }
    
    for(i = 0;i < device_count; i++){
	s+= sprintf(s, "affinity     %d    %s\n", i, node_affinity_string[gpu_affinity_table[i]]);
    }
    
    file = fopen(output_file, "w");
    if (file == NULL){
	printf("ERROR: opening file %s for write failed\n");
	exit(1);
    }
    fprintf(file, "%s", filebuf);
    fclose(file);    
}

void 
usage(char** argv)
{
    printf("This utility detects the GPU/CPU affinity by performing exhaustive bandwidth tests\n");
    printf("Usage: %s [options]\n", argv[0]);
    printf("options\n");
    printf("-v                 Verbose, print out the bandwidth table\n");
    printf("-vv                More verbose message, print out all measurements and related information\n");
    printf("-o <output_file>   Output file for GPU/CPU affinity. The file is created if it does not exists.\n");
    printf("                   If the file exists, the old affinity entries is removed. Other content in the file remains the same\n");	   
    printf("--help             This message\n");

    exit(1);
}

int
main(int argc, char** argv)
{
    char* output_file = NULL;    
    int i;
    for (i =1;i < argc; i++){
	
	if( strcmp(argv[i], "--help")== 0){
	    usage(argv);
	}
	
	if( strcmp(argv[i], "-o") == 0){
	    if (i+1 >= argc){
		usage(argv);
	    }
	    output_file =  argv[i+1];
	    i++;
	    continue;
	}

	if( strcmp(argv[i], "-v") == 0){
	    verbose = 1;
	    continue;
	}	
	
	if( strcmp(argv[i], "-vv") == 0){
	    verbose = 2;
	    continue;
	}		
	

	printf("ERROR: invalid option\n");
	usage(argv);
	
    }
    

    int nodes;
    
    //nodes = numa_num_configured_nodes();
    nodes = numa_max_node() +1;
    if (nodes <= 0){
	printf("ERROR: no nodes found\n");
	exit(1);
    }
    
    int device_count;

    cudaGetDeviceCount(&device_count); CUERR;
    
    
    double* bandwidth_table;
    bandwidth_table = (double*) malloc(nodes*device_count*sizeof(double));
    if (bandwidth_table == NULL){
	printf("ERROR: malloc failed for bandwidth table\n");
	exit(1);
    }
    memset(bandwidth_table, 0, nodes*device_count*sizeof(double));

    int* gpu_affinity_table;
    gpu_affinity_table = (int*)malloc(device_count*sizeof(int));
    if (gpu_affinity_table == NULL){
	printf("ERROR: malloc failed for gpu_affinity_table\n");
	exit(1);
    }
    memset(gpu_affinity_table, 0, device_count* sizeof(int));

    char node_affinity_string[MAX_NODES][MAX_STRING_LEN];
    
    int j;
#if (LIBNUMA_API_VERSION >=2)
     struct bitmask bitmask_buf;
     unsigned long mask;
     bitmask_buf.size = sizeof(unsigned long);
     bitmask_buf.maskp = &mask;
#else
    char bitmask_buf[64];
#endif

    for(i=0; i < nodes;i++){
	int rc;
#if (LIBNUMA_API_VERSION >=2)
	rc =numa_node_to_cpus(i, &bitmask_buf);
#else
	rc =numa_node_to_cpus(i, (unsigned long*)&bitmask_buf, sizeof(bitmask_buf));
#endif
	if (rc != 0){
	    printf("Error: changing node %d to cpu mask failed(errno=%d)\n", i, errno);
	    exit(1);
	}
#if (LIBNUMA_API_VERSION >=2)
	VERBOSE_PRINT("node %d has mask: %x\n", i,*(unsigned long*)bitmask_buf.maskp);
#else
	VERBOSE_PRINT("node %d has mask: %x\n", i,*(unsigned long*)&bitmask_buf);
#endif
	
	for(j = 0; j < device_count; j++){
	    double bandwidth  = bandwidth_test(i, (unsigned long*)&bitmask_buf, j);	    
	    bandwidth_table[j*nodes + i] = bandwidth;
	}
	int cpu = 0;
	int firsttime= 1;
	char* s = node_affinity_string[i];
#if (LIBNUMA_API_VERSION >=2)
        unsigned long bitmask = *(unsigned long*)bitmask_buf.maskp;
#else
        unsigned long bitmask = *(unsigned long*)&bitmask_buf;
#endif
	for(j =0; j < sizeof(unsigned long)*8; j++){
	    if (bitmask & 1){
		if (firsttime){
		    s += sprintf(s, "%d", cpu);
		    firsttime =0;
		}else{
		    s+=sprintf(s,",%d", cpu);
		}
	    }
	    bitmask = bitmask >> 1;
	    cpu++;
	}
	//sprintf(s, "\n");
	VERBOSE_PRINT("node_affiity_string[%d]: %s\n", i, node_affinity_string[i]);
	
    }
    
    
    display_bandwidth_table(bandwidth_table, nodes, device_count);
    
    process_bandwidth_table(bandwidth_table, nodes, device_count, gpu_affinity_table);
    
    if (!output_file){
	printf("#affinity  <GPU>  nodeid   cpu1,cpu2,...\n");
	for(i = 0;i < device_count; i++){
	    printf("affinity     %d      %d       %s\n", i, gpu_affinity_table[i], node_affinity_string[gpu_affinity_table[i]]);
	}
    }else{
	output_to_file(node_affinity_string, gpu_affinity_table, device_count, output_file);
    }
    free(bandwidth_table);
    free(gpu_affinity_table);
    
    return 0;

}



