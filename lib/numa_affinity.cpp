
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
#include <quda_internal.h>

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
    warningQuda("Bad argument");
    return -1;
  }

  if(strstr(str, "-") != NULL){
    //a range
    int low_core, high_core;
    if (sscanf(str,"%d-%d",&low_core, &high_core) != 2){
      warningQuda("Range scan failed");
      return -1;
    }
    if(*sub_ncores <  high_core-low_core +1){
      warningQuda("Not enough space in sub_list");
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
      warningQuda("Wrong format for core number");
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
    warningQuda("Bad argument");
    return  -1;
  }

  char str[256];
  strncpy(str, _str, sizeof(str));

  int left_space = *ncores;
  int tot_cores = 0;

  char* item = strtok(str, ",");
  if(item == NULL){
    warningQuda("Invalid string format (%s)", str);
    return -1;
  }
  
  do {
    int sub_ncores = left_space;
    int* sub_list = list + tot_cores;
    
    int rc = process_core_string_item(item, sub_list, &sub_ncores);
    if(rc <0){
      warningQuda("Processing item (%s) failed", item);
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
  
  my_line= (char *) safe_malloc(nbytes +1);
  
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
    //printfQuda("Warning: opening file %s failed\n", pci_bus_info_path);
    host_free(my_line);
    fclose(nvidia_info);
    return -1;
  }
  
  while (!feof(pci_bus_info)){
    if ( -1 == getline(&my_line, &nbytes, pci_bus_info)){
      break;
    } else{
      int rc = process_core_string_list(my_line, cpu_cores, ncores);
      if(rc < 0){
	warningQuda("Failed to process the line \"%s\"", my_line);
	host_free(my_line);
	fclose(nvidia_info);
	return  -1;
      }
    }
  }
  
  host_free(my_line);
  return 0;
}

int 
setNumaAffinity(int devid)
{
  int cpu_cores[128];
  int ncores=128;
  int rc = getNumaAffinity(devid, cpu_cores, &ncores);
  if(rc != 0){
    warningQuda("Failed to determine NUMA affinity for device %d (possibly not applicable)", devid);
    return 1;
  }
  int which = devid % ncores;
  printfQuda("Setting NUMA affinity for device %d to CPU core %d\n", devid, cpu_cores[which]);
/*
  for(int i=0;i < ncores;i++){
   if (i != which ) continue;
    printfQuda("%d", cpu_cores[i]);
    if((i+1) < ncores){
      printfQuda(",");
    }
  }
  printfQuda("\n");
  */

  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  
  for(int i=0;i < ncores;i++){
    if( i != which) continue;
    CPU_SET(cpu_cores[i], &cpu_set);
  }
  
  rc = sched_setaffinity(0, sizeof(cpu_set_t), &cpu_set);
  if (rc != 0){
    warningQuda("Failed to enforce NUMA affinity (probably due to lack of kernel support)");
    return -1;
  }
  
  
  return 0;
}

