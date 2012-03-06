
/* Originally from Galen Arnold, NCSA arnoldg@ncsa.illinois.edu 
 * modified by Guochun Shi
 *
 */
#undef _GNU_SOURCE
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sched.h>
#include <unistd.h>
#include <string.h>
#include <numa_affinity.h>
#include <util_quda.h>

static int 
process_core_string_item(const char* str, int* sub_list, int* sub_ncores)
{
  /* assume the input format is one of the following two
   * 1. a number only, e.g. 5
   * 2. a range, e.g 4-6, which means three numbers 4,5,6
   * return a list of numbers in @sub_list and and the total numbers
   * in @sub_ncores
   */
  int i;
  if(str == NULL || sub_list == NULL || sub_ncores == NULL ||
     *sub_ncores <= 0){
    printfQuda("Warning: Wrong parameters in function %s!\n", __FUNCTION__);
    return -1;
  }

  if(strstr(str, "-") != NULL){
    //a range
    int low_core, high_core;
    if (sscanf(str,"%d-%d",&low_core, &high_core) != 2){
      printfQuda("Warning: range scan failed\n");
      return -1;
    }
    if(*sub_ncores <  high_core-low_core +1){
      printfQuda("Warning: not enough space in sub_list\n");
      return -1;
    }
    
    for(i = 0; i < high_core-low_core +1; i++){
      sub_list[i] = i + low_core;
    }
    *sub_ncores =  high_core - low_core +1;

  }else{
    //a number
    int core;
    if (sscanf(str, "%d", &core) != 1){
      printfQuda("Warning: wrong format for core number\n");
      return -1;
    }
    sub_list[0] = core;
    *sub_ncores   =1;
  }
  return 0;
}

static int
process_core_string_list(const char* _str, int* list, int* ncores)
{
  /* The input string @str should be separated by comma, and each item can be 
   * either a number or a range (see the comments in process_core_string_item 
   * function)
   *
   */

  if(_str == NULL || list == NULL || ncores == NULL
     || *ncores <= 0){
    printfQuda("Warning: Invalid arguments in function %s\n", __FUNCTION__ );
    return  -1;
  }

  char str[256];
  strncpy(str, _str, sizeof(str));

  int left_space = *ncores;
  int tot_cores = 0;

  char* item = strtok(str, ",");
  if(item == NULL){
    printfQuda("ERROR: Invalid string format(%s)\n", str);
    return -1;
  }
  
  do {
    int sub_ncores = left_space;
    int* sub_list = list + tot_cores;
    
    int rc = process_core_string_item(item, sub_list, &sub_ncores);
    if(rc <0){
      printfQuda("Warning: processing item(%s) failed\n", item);
      return -1;
    }

    tot_cores += sub_ncores;
    left_space -= sub_ncores;

    item = strtok(NULL, ",");
  }while( item != NULL);

  *ncores = tot_cores;
  return 0;
}


static int 
getNumaAffinity(int my_gpu, int *cpu_cores, int* ncores)
{
  FILE *nvidia_info, *pci_bus_info;
  size_t nbytes = 255;
  char *my_line;
  char nvidia_info_path[255], pci_bus_info_path[255];
  char bus_info[255];
  
  // the nvidia driver populates this path for each gpu
  sprintf(nvidia_info_path,"/proc/driver/nvidia/gpus/%d/information", my_gpu);
  nvidia_info= fopen(nvidia_info_path,"r");
  if (nvidia_info == NULL){
    return -1;
  }
  
  my_line= (char *) malloc(nbytes +1);
  if (my_line == NULL){ 
    errorQuda("Error: allocating memory for my_line failed"); 
  }
  
  while (!feof(nvidia_info)){
    if ( -1 == getline(&my_line, &nbytes, nvidia_info)){
      break;
    }else{
      // the first 7 char of the Bus Location will lead to the corresponding
      // path under /sys/class/pci_bus/  , cpulistaffinity showing cores on that
      // bus is located there
      if ( 1 == sscanf(my_line,"Bus Location: %s", bus_info )){
	sprintf(pci_bus_info_path,"/sys/class/pci_bus/%.7s/cpulistaffinity",
		bus_info);
      }
    }
  }
  // open the cpulistaffinity file on the pci_bus for "my_gpu"
  pci_bus_info= fopen(pci_bus_info_path,"r");
  if (pci_bus_info == NULL){
    printfQuda("Warning: opening file %s failed\n", pci_bus_info_path);
    free(my_line);
    fclose(nvidia_info);
    return -1;
  }
  
  while (!feof(pci_bus_info)){
    if ( -1 == getline(&my_line, &nbytes, pci_bus_info)){
      break;
    } else{
      int rc = process_core_string_list(my_line, cpu_cores, ncores);
      if(rc < 0){
	printfQuda("Warning:%s: processing the line (%s) failed\n", __FUNCTION__, my_line);
	free(my_line);
	fclose(nvidia_info);
	return  -1;
      }
    }
  }
  
  free(my_line);
  return(0);
}


int 
setNumaAffinity(int devid)
{
  int cpu_cores[128];
  int ncores=128;
  int rc = getNumaAffinity(devid, cpu_cores, &ncores);
  if(rc != 0){
    printfQuda("Warning: quda getting affinity for device %d failed\n", devid);
    return 1;
  }
  printfQuda("GPU: %d, Setting to affinity cpu cores: ", devid);
  for(int i=0;i < ncores;i++){
    printfQuda("%d", cpu_cores[i]);
    if((i+1) < ncores){
      printfQuda(",");
    }
  }
  printfQuda("\n");
  
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  
  for(int i=0;i < ncores;i++){
    CPU_SET(cpu_cores[i], &cpu_set);
  }
  
  rc = sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
  if (rc != 0){
    printfQuda("Warning: quda settting affinity failed\n");
    return -1;
  }
  
  
  return 0;
}
