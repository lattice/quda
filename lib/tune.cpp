#include <tune_quda.h>
#include <comm_quda.h>
#include <quda.h> // for QUDA_VERSION_STRING
#include <sys/stat.h> // for stat()
#include <fcntl.h>
#include <cfloat> // for FLT_MAX
#include <ctime>
#include <fstream>
#include <typeinfo>
#include <map>

static const std::string quda_hash = QUDA_HASH; // defined in lib/Makefile
static std::string resource_path;
static std::map<TuneKey, TuneParam> tunecache;
static size_t initial_cache_size = 0;

#define STR_(x) #x
#define STR(x) STR_(x)
static const std::string quda_version = STR(QUDA_VERSION_MAJOR) "." STR(QUDA_VERSION_MINOR) "." STR(QUDA_VERSION_SUBMINOR);
#undef STR
#undef STR_


/**
 * Deserialize tunecache from an istream, useful for reading a file or receiving from other nodes.
 */
static void deserializeTuneCache(std::istream &in)
{
  std::string line;
  std::stringstream ls;
  TuneKey key;
  TuneParam param;

  while (in.good()) {
    getline(in, line);
    if (!line.length()) continue; // skip blank lines (e.g., at end of file)
    ls.clear();
    ls.str(line);
    ls >> key.volume >> key.name >> key.aux >> param.block.x >> param.block.y >> param.block.z;
    ls >> param.grid.x >> param.grid.y >> param.grid.z >> param.shared_bytes;
    ls.ignore(1); // throw away tab before comment
    getline(ls, param.comment); // assume anything remaining on the line is a comment
    param.comment += "\n"; // our convention is to include the newline, since ctime() likes to do this
    tunecache[key] = param;
  }
}


/**
 * Serialize tunecache to an ostream, useful for writing to a file or sending to other nodes.
 */
static void serializeTuneCache(std::ostream &out)
{
  std::map<TuneKey, TuneParam>::iterator entry;

  for (entry = tunecache.begin(); entry != tunecache.end(); entry++) {
    TuneKey key = entry->first;
    TuneParam param = entry->second;

    out << key.volume << "\t" << key.name << "\t" << key.aux << "\t";
    out << param.block.x << "\t" << param.block.y << "\t" << param.block.z << "\t";
    out << param.grid.x << "\t" << param.grid.y << "\t" << param.grid.z << "\t";
    out << param.shared_bytes << "\t" << param.comment; // param.comment ends with a newline
  }
}


/**
 * Distribute the tunecache from node 0 to all other nodes.
 */
static void broadcastTuneCache()
{
#ifdef MULTI_GPU

  std::stringstream serialized;
  size_t size;

  if (comm_rank() == 0) {
    serializeTuneCache(serialized);
    size = serialized.str().length();
  }
  comm_broadcast(&size, sizeof(size_t));

  if (size > 0) {
    if (comm_rank() == 0) {
      comm_broadcast(const_cast<char *>(serialized.str().c_str()), size);
    } else {
      char *serstr = new char[size+1];
      comm_broadcast(serstr, size);
      serstr[size] ='\0'; // null-terminate
      serialized.str(serstr);
      deserializeTuneCache(serialized);
      delete[] serstr;
    }
  }
#endif
}


/*
 * Read tunecache from disk.
 */
void loadTuneCache(QudaVerbosity verbosity)
{
  char *path;
  struct stat pstat;
  std::string cache_path, line, token;
  std::ifstream cache_file;
  std::stringstream ls;

  path = getenv("QUDA_RESOURCE_PATH");
  if (!path) {
    warningQuda("Environment variable QUDA_RESOURCE_PATH is not set.");
    warningQuda("Caching of tuned parameters will be disabled.");
    return;
  } else if (stat(path, &pstat) || !S_ISDIR(pstat.st_mode)) {
    warningQuda("The path \"%s\" specified by QUDA_RESOURCE_PATH does not exist or is not a directory.", path); 
    warningQuda("Caching of tuned parameters will be disabled.");
    return;
  } else {
    resource_path = path;
  }

#ifdef MULTI_GPU
  if (comm_rank() == 0) {
#endif

    cache_path = resource_path;
    cache_path += "/tunecache.tsv";
    cache_file.open(cache_path.c_str());

    if (cache_file) {

      if (!cache_file.good()) errorQuda("Bad format in %s", cache_path.c_str());
      getline(cache_file, line);
      ls.str(line);
      ls >> token;
      if (token.compare("tunecache")) errorQuda("Bad format in %s", cache_path.c_str());
      ls >> token;
      if (token.compare(quda_version)) errorQuda("Cache file %s does not match current QUDA version", cache_path.c_str());
      ls >> token;
      if (token.compare(quda_hash)) warningQuda("Cache file %s does not match current QUDA build", cache_path.c_str());
      
      if (!cache_file.good()) errorQuda("Bad format in %s", cache_path.c_str());
      getline(cache_file, line); // eat the blank line
      
      if (!cache_file.good()) errorQuda("Bad format in %s", cache_path.c_str());
      getline(cache_file, line); // eat the description line
      
      deserializeTuneCache(cache_file);
      cache_file.close();      
      initial_cache_size = tunecache.size();

      if (verbosity >= QUDA_SUMMARIZE) {
	printfQuda("Loaded %d sets of cached parameters from %s\n", static_cast<int>(initial_cache_size), cache_path.c_str());
      }
      
    } else {
      warningQuda("Cache file not found.  All kernels will be re-tuned (if tuning is enabled).");
    }

#ifdef MULTI_GPU
  }
#endif

  broadcastTuneCache();
}


/**
 * Write tunecache to disk.
 */
void saveTuneCache(QudaVerbosity verbosity)
{
  time_t now;
  int lock_handle;
  std::string lock_path, cache_path;
  std::ofstream cache_file;

  if (resource_path.empty()) return;

  //FIXME: We should really check to see if any nodes have tuned a kernel that was not also tuned on node 0, since as things
  //       stand, the corresponding launch parameters would never get cached to disk in this situation.  This will come up if we
  //       ever support different subvolumes per GPU (as might be convenient for lattice volumes that don't divide evenly).

#ifdef MULTI_GPU
  if (comm_rank() == 0) {
#endif

    if (tunecache.size() == initial_cache_size) return;

    // Acquire lock.  Note that this is only robust if the filesystem supports flock() semantics, which is true for
    // NFS on recent versions of linux but not Lustre by default (unless the filesystem was mounted with "-o flock").
    lock_path = resource_path + "/tunecache.lock";
    lock_handle = open(lock_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (lock_handle == -1) {
      warningQuda("Unable to lock cache file.  Tuned launch parameters will not be cached to disk.  "
                  "If you are certain that no other instances of QUDA are accessing this filesystem, "
                  "please manually remove %s", lock_path.c_str());
      return;
    }
    char msg[] = "If no instances of applications using QUDA are running,\n"
                 "this lock file shouldn't be here and is safe to delete.";
    int stat = write(lock_handle, msg, sizeof(msg)); // check status to avoid compiler warning
    if (stat == -1) warningQuda("Unable to write to lock file for some bizarre reason");

    cache_path = resource_path + "/tunecache.tsv";
    cache_file.open(cache_path.c_str());
    
    if (verbosity >= QUDA_SUMMARIZE) {
      printfQuda("Saving %d sets of cached parameters to %s\n", static_cast<int>(tunecache.size()), cache_path.c_str());
    }
    
    time(&now);
    cache_file << "tunecache\t" << quda_version << "\t" << quda_hash << "\t# Last updated " << ctime(&now) << std::endl;
    cache_file << "volume\tname\taux\tblock.x\tblock.y\tblock.z\tgrid.x\tgrid.y\tgrid.z\tshared_bytes\tcomment" << std::endl;
    serializeTuneCache(cache_file);
    cache_file.close();

    // Release lock.
    close(lock_handle);
    remove(lock_path.c_str());

#ifdef MULTI_GPU
  }
#endif
}


/**
 * Return the optimal launch parameters for a given kernel, either by retrieving them from tunecache or autotuning
 * on the spot.
 */
TuneParam tuneLaunch(Tunable &tunable, QudaTune enabled, QudaVerbosity verbosity)
{
  static bool tuning = false; // tuning in progress?
  static const Tunable *active_tunable; // for error checking
  static TuneParam param;

  TuneParam best_param;
  cudaError_t error;
  cudaEvent_t start, end;
  float elapsed_time, best_time;
  time_t now;

  const TuneKey key = tunable.tuneKey();

  if (enabled == QUDA_TUNE_NO) {
    tunable.defaultTuneParam(param);
  } else if (tunecache.count(key)) {
    param = tunecache[key];
  } else if (!tuning) {

    tuning = true;
    active_tunable = &tunable;
    best_time = FLT_MAX;
    tunable.preTune();

    cudaEventCreate(&start);
    cudaEventCreate(&end);

    if (verbosity >= QUDA_DEBUG_VERBOSE) {
      printfQuda("Tuning %s with %s at vol=%s\n", key.name.c_str(), key.aux.c_str(), key.volume.c_str());
    }

    tunable.initTuneParam(param);
    while (tuning) {
      cudaThreadSynchronize();
      cudaGetLastError(); // clear error counter
      cudaEventRecord(start, 0);
      for (int i=0; i<tunable.tuningIter(); i++) {
	tunable.apply(0);  // calls tuneLaunch() again, which simply returns the currently active param
      }
      cudaEventRecord(end, 0);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&elapsed_time, start, end);
      cudaThreadSynchronize();
      error = cudaGetLastError();
      elapsed_time /= (1e3 * tunable.tuningIter());
      if ((elapsed_time < best_time) && (error == cudaSuccess)) {
	best_time = elapsed_time;
	best_param = param;
      }
      if ((verbosity >= QUDA_DEBUG_VERBOSE)) {
	if (error == cudaSuccess)
	  printfQuda("    %s gives %s\n", tunable.paramString(param).c_str(), 
		     tunable.perfString(elapsed_time).c_str());
	else 
	  printfQuda("    %s gives %s\n", tunable.paramString(param).c_str(), cudaGetErrorString(error));
      }
      tuning = tunable.advanceTuneParam(param);
    }

    if (best_time == FLT_MAX) {
      errorQuda("Auto-tuning failed for %s with %s at vol=%s", key.name.c_str(), key.aux.c_str(), key.volume.c_str());
    }
    if (verbosity >= QUDA_VERBOSE) {
      printfQuda("Tuned %s giving %s", tunable.paramString(best_param).c_str(), tunable.perfString(best_time).c_str());
      printfQuda(" for %s with %s\n", key.name.c_str(), key.aux.c_str());
    }
    time(&now);
    best_param.comment = "# " + tunable.perfString(best_time) + ", tuned ";
    best_param.comment += ctime(&now); // includes a newline

    cudaEventDestroy(start);
    cudaEventDestroy(end);

    tunable.postTune();
    param = best_param;
    tunecache[key] = best_param;

  } else if (&tunable != active_tunable) {
    errorQuda("Unexpected call to tuneLaunch() in %s::apply()", typeid(tunable).name());
  }

  return param;
}
