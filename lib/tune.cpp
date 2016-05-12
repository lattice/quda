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
#include <unistd.h>

#include <deque>
#include <queue>
#include <functional>
#ifdef PTHREADS
#include <pthread.h>
#endif

//#define LAUNCH_TIMER
extern char* gitversion;

namespace quda { static TuneKey last_key; }

// intentionally leave this outside of the namespace for now
quda::TuneKey getLastTuneKey() { return quda::last_key; }

namespace quda {
  typedef std::map<TuneKey, TuneParam> map;

  static const std::string quda_hash = QUDA_HASH; // defined in lib/Makefile
  static std::string resource_path;
  static map tunecache;
  static map::iterator it;
  static size_t initial_cache_size = 0;


#define STR_(x) #x
#define STR(x) STR_(x)
  static const std::string quda_version = STR(QUDA_VERSION_MAJOR) "." STR(QUDA_VERSION_MINOR) "." STR(QUDA_VERSION_SUBMINOR);
#undef STR
#undef STR_

  static bool profile_count = true;

  void disableProfileCount() { profile_count = false; }
  void enableProfileCount() { profile_count = true; }

  const map& getTuneCache() { return tunecache; }


  /**
   * Deserialize tunecache from an istream, useful for reading a file or receiving from other nodes.
   */
  static void deserializeTuneCache(std::istream &in)
  {
    std::string line;
    std::stringstream ls;

    TuneKey key;
    TuneParam param;

    std::string v;
    std::string n;
    std::string a;

    int check;

    while (in.good()) {
      getline(in, line);
      if (!line.length()) continue; // skip blank lines (e.g., at end of file)
      ls.clear();
      ls.str(line);
      ls >> v >> n >> a >> param.block.x >> param.block.y >> param.block.z;
      check = snprintf(key.volume, key.volume_n, "%s", v.c_str());
      if (check < 0 || check >= key.volume_n) errorQuda("Error writing volume string");
      check = snprintf(key.name, key.name_n, "%s", n.c_str());
      if (check < 0 || check >= key.name_n) errorQuda("Error writing name string");
      check = snprintf(key.aux, key.aux_n, "%s", a.c_str());
      if (check < 0 || check >= key.aux_n) errorQuda("Error writing aux string");
      ls >> param.grid.x >> param.grid.y >> param.grid.z >> param.shared_bytes >> param.aux.x >> param.aux.y >> param.aux.z >> param.time;
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
    map::iterator entry;

    for (entry = tunecache.begin(); entry != tunecache.end(); entry++) {
      TuneKey key = entry->first;
      TuneParam param = entry->second;

      out << std::setw(16) << key.volume << "\t" << key.name << "\t" << key.aux << "\t";
      out << param.block.x << "\t" << param.block.y << "\t" << param.block.z << "\t";
      out << param.grid.x << "\t" << param.grid.y << "\t" << param.grid.z << "\t";
      out << param.shared_bytes << "\t" << param.aux.x << "\t" << param.aux.y << "\t" << param.aux.z << "\t";
      out << param.time << "\t" << param.comment; // param.comment ends with a newline
    }
  }


  template <class T>
  struct less_significant : std::binary_function<T,T,bool> {
    inline bool operator()(const T &lhs, const T &rhs) {
      return lhs.second.time * lhs.second.n_calls < rhs.second.time * rhs.second.n_calls;
    }
  };

  /**
   * Serialize tunecache to an ostream, useful for writing to a file or sending to other nodes.
   */
  static void serializeProfile(std::ostream &out, std::ostream &async_out)
  {
    map::iterator entry;
    double total_time = 0.0;
    double async_total_time = 0.0;

    // first let's sort the entries in decreasing order of significance
    typedef std::pair<TuneKey, TuneParam> profile_t;
    typedef std::priority_queue<profile_t, std::deque<profile_t>, less_significant<profile_t> > queue_t;
    queue_t q(tunecache.begin(), tunecache.end());

    // now compute total time spent in kernels so we can give each kernel a significance
    for (entry = tunecache.begin(); entry != tunecache.end(); entry++) {
      TuneKey key = entry->first;
      TuneParam param = entry->second;

      char tmp[7] = { };
      strncpy(tmp, key.aux, 6);
      bool is_policy = strcmp(tmp, "policy") == 0 ? true : false;
      if (param.n_calls > 0 && !is_policy) total_time += param.n_calls * param.time;
      if (param.n_calls > 0 && is_policy) async_total_time += param.n_calls * param.time;
    }


    while ( !q.empty() ) {
      TuneKey key = q.top().first;
      TuneParam param = q.top().second;

      char tmp[7] = { };
      strncpy(tmp, key.aux, 6);
      bool is_policy = strcmp(tmp, "policy") == 0 ? true : false;

      // synchronous profile
      if (param.n_calls > 0 && !is_policy) {
	double time = param.n_calls * param.time;

	out << std::setw(12) << param.n_calls * param.time << "\t" << std::setw(12) << (time / total_time) * 100 << "\t";
	out << std::setw(12) << param.n_calls << "\t" << std::setw(12) << param.time << "\t" << std::setw(16) << key.volume << "\t";
	out << key.name << "\t" << key.aux << "\t" << param.comment; // param.comment ends with a newline
      }

      // async policy profile
      if (param.n_calls > 0 && is_policy) {
	double time = param.n_calls * param.time;

	async_out << std::setw(12) << param.n_calls * param.time << "\t" << std::setw(12) << (time / async_total_time) * 100 << "\t";
	async_out << std::setw(12) << param.n_calls << "\t" << std::setw(12) << param.time << "\t" << std::setw(16) << key.volume << "\t";
	async_out << key.name << "\t" << key.aux << "\t" << param.comment; // param.comment ends with a newline
      }

      q.pop();
    }

    out << std::endl << "# Total time spent in kernels = " << total_time << " seconds" << std::endl;
    async_out << std::endl << "# Total time spent in asynchronous execution = " << async_total_time << " seconds" << std::endl;
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
	if (token.compare(quda_version)) errorQuda("Cache file %s does not match current QUDA version. \nPlease delete this file or set the QUDA_RESOURCE_PATH environment variable to point to a new path.", cache_path.c_str());
	ls >> token;
#ifdef GITVERSION
	if (token.compare(gitversion)) errorQuda("Cache file %s does not match current QUDA version. \nPlease delete this file or set the QUDA_RESOURCE_PATH environment variable to point to a new path.", cache_path.c_str());
#else
	if (token.compare(quda_version)) errorQuda("Cache file %s does not match current QUDA version. \nPlease delete this file or set the QUDA_RESOURCE_PATH environment variable to point to a new path.", cache_path.c_str());
#endif
	ls >> token;
	if (token.compare(quda_hash)) errorQuda("Cache file %s does not match current QUDA build. \nPlease delete this file or set the QUDA_RESOURCE_PATH environment variable to point to a new path.", cache_path.c_str());


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
      cache_file << "tunecache\t" << quda_version;
#ifdef GITVERSION
       cache_file << "\t" << gitversion;
#else
       cache_file << "\t" << quda_version;
#endif
      cache_file << "\t" << quda_hash << "\t# Last updated " << ctime(&now) << std::endl;
      cache_file << std::setw(16) << "volume" << "\tname\taux\tblock.x\tblock.y\tblock.z\tgrid.x\tgrid.y\tgrid.z\tshared_bytes\taux.x\taux.y\taux.z\ttime\tcomment" << std::endl;
      serializeTuneCache(cache_file);
      cache_file.close();

      // Release lock.
      close(lock_handle);
      remove(lock_path.c_str());

      initial_cache_size = tunecache.size();

#ifdef MULTI_GPU
    }
#endif
  }


  /**
   * Write profile to disk.
   */
  void saveProfile(QudaVerbosity verbosity)
  {
    time_t now;
    int lock_handle;
    std::string lock_path, profile_path, async_profile_path;
    std::ofstream profile_file, async_profile_file;

    if (resource_path.empty()) return;

#ifdef MULTI_GPU
    if (comm_rank() == 0) {
#endif

      // Acquire lock.  Note that this is only robust if the filesystem supports flock() semantics, which is true for
      // NFS on recent versions of linux but not Lustre by default (unless the filesystem was mounted with "-o flock").
      lock_path = resource_path + "/profile.lock";
      lock_handle = open(lock_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
      if (lock_handle == -1) {
	warningQuda("Unable to lock profile file.  Profile will not be saved to disk.  "
		    "If you are certain that no other instances of QUDA are accessing this filesystem, "
		    "please manually remove %s", lock_path.c_str());
	return;
      }
      char msg[] = "If no instances of applications using QUDA are running,\n"
	"this lock file shouldn't be here and is safe to delete.";
      int stat = write(lock_handle, msg, sizeof(msg)); // check status to avoid compiler warning
      if (stat == -1) warningQuda("Unable to write to lock file for some bizarre reason");

      char *profile_fname = getenv("QUDA_PROFILE_OUTPUT_BASE");

      if (!profile_fname) {
	warningQuda("Environment variable QUDA_PROFILE_OUTPUT_BASE is not set; writing to profile.tsv and profile_async.tsv");
	profile_path = resource_path + "/profile.tsv";
	async_profile_path = resource_path + "/profile_async.tsv";
      } else {
	profile_path = resource_path + "/" + profile_fname + ".tsv";
	async_profile_path = resource_path + "/" + profile_fname + "_async.tsv";
      }
      profile_file.open(profile_path.c_str());
      async_profile_file.open(async_profile_path.c_str());

      if (verbosity >= QUDA_SUMMARIZE) {
	// compute number of non-zero entries that will be output in the profile
	int n_entry = 0;
	int n_policy = 0;
	for (map::iterator entry = tunecache.begin(); entry != tunecache.end(); entry++) {
	  // if a policy entry, then we can ignore
	  char tmp[7] = { };
	  strncpy(tmp, entry->first.aux, 6);
	  TuneParam param = entry->second;
	  bool is_policy = strcmp(tmp, "policy") == 0 ? true : false;
	  if (param.n_calls > 0 && !is_policy) n_entry++;
	  if (param.n_calls > 0 && is_policy) n_policy++;
	}

	printfQuda("Saving %d sets of cached parameters to %s\n", n_entry, profile_path.c_str());
	printfQuda("Saving %d sets of cached profiles to %s\n", n_policy, async_profile_path.c_str());
      }

      time(&now);

      profile_file << "profile\t" << quda_version;
#ifdef GITVERSION
      profile_file << "\t" << gitversion;
#else
      profile_file << "\t" << quda_version;
#endif
      profile_file << "\t" << quda_hash << "\t# Last updated " << ctime(&now) << std::endl;
      profile_file << std::setw(12) << "total time" << "\t" << std::setw(12) << "percentage" << "\t" << std::setw(12) << "calls" << "\t" << std::setw(12) << "time / call" << "\t" << std::setw(16) << "volume" << "\tname\taux\tcomment" << std::endl;

      async_profile_file << "profile\t" << quda_version;
#ifdef GITVERSION
      async_profile_file << "\t" << gitversion;
#else
      async_profile_file << "\t" << quda_version;
#endif
      async_profile_file << "\t" << quda_hash << "\t# Last updated " << ctime(&now) << std::endl;
      async_profile_file << std::setw(12) << "total time" << "\t" << std::setw(12) << "percentage" << "\t" << std::setw(12) << "calls" << "\t" << std::setw(12) << "time / call" << "\t" << std::setw(16) << "volume" << "\tname\taux\tcomment" << std::endl;

      serializeProfile(profile_file, async_profile_file);

      profile_file.close();
      async_profile_file.close();

      // Release lock.
      close(lock_handle);
      remove(lock_path.c_str());

#ifdef MULTI_GPU
    }
#endif
  }


  static TimeProfile launchTimer("tuneLaunch");

//  static int tally = 0;

  /**
   * Return the optimal launch parameters for a given kernel, either
   * by retrieving them from tunecache or autotuning on the spot.
   */
  TuneParam& tuneLaunch(Tunable &tunable, QudaTune enabled, QudaVerbosity verbosity)
  {
#ifdef PTHREADS // tuning should be performed serially
//  pthread_mutex_lock(&pthread_mutex);
//  tally++;
#endif

#ifdef LAUNCH_TIMER
    launchTimer.TPSTART(QUDA_PROFILE_TOTAL);
    launchTimer.TPSTART(QUDA_PROFILE_INIT);
#endif

    const TuneKey key = tunable.tuneKey();
    last_key = key;
    static TuneParam param;

#ifdef LAUNCH_TIMER
    launchTimer.TPSTOP(QUDA_PROFILE_INIT);
    launchTimer.TPSTART(QUDA_PROFILE_PREAMBLE);
#endif

    static bool tuning = false; // tuning in progress?
    static const Tunable *active_tunable; // for error checking

    // first check if we have the tuned value and return if we have it
    //if (enabled == QUDA_TUNE_YES && tunecache.count(key)) {

    it = tunecache.find(key);
    if (enabled == QUDA_TUNE_YES && it != tunecache.end()) {

#ifdef LAUNCH_TIMER
      launchTimer.TPSTOP(QUDA_PROFILE_PREAMBLE);
      launchTimer.TPSTART(QUDA_PROFILE_COMPUTE);
#endif

      TuneParam &param = it->second;

#ifdef LAUNCH_TIMER
      launchTimer.TPSTOP(QUDA_PROFILE_COMPUTE);
      launchTimer.TPSTART(QUDA_PROFILE_EPILOGUE);
#endif

      tunable.checkLaunchParam(param);

#ifdef LAUNCH_TIMER
      launchTimer.TPSTOP(QUDA_PROFILE_EPILOGUE);
      launchTimer.TPSTOP(QUDA_PROFILE_TOTAL);
#endif

#ifdef PTHREADS
      //pthread_mutex_unlock(&pthread_mutex);
      //tally--;
      //printfQuda("pthread_mutex_unlock a complete %d\n",tally);
#endif
      // we could be tuning outside of the current scope
      if (!tuning && profile_count) param.n_calls++;

      return param;
    }

#ifdef LAUNCH_TIMER
    launchTimer.TPSTOP(QUDA_PROFILE_PREAMBLE);
    launchTimer.TPSTOP(QUDA_PROFILE_TOTAL);
#endif


    // We must switch off the global sum when tuning in case of process divergence
    bool reduceState = commGlobalReduction();
    commGlobalReductionSet(false);

    if (enabled == QUDA_TUNE_NO) {
      tunable.defaultTuneParam(param);
      tunable.checkLaunchParam(param);
    } else if (!tuning) {

      TuneParam best_param;
      cudaError_t error;
      cudaEvent_t start, end;
      float elapsed_time, best_time;
      time_t now;

      tuning = true;
      active_tunable = &tunable;
      best_time = FLT_MAX;

      if (verbosity >= QUDA_DEBUG_VERBOSE) printfQuda("PreTune %s\n", key.name);
      tunable.preTune();

      cudaEventCreate(&start);
      cudaEventCreate(&end);

      if (verbosity >= QUDA_DEBUG_VERBOSE) {
	printfQuda("Tuning %s with %s at vol=%s\n", key.name, key.aux, key.volume);
      }

      tunable.initTuneParam(param);
      while (tuning) {
	cudaDeviceSynchronize();
	cudaGetLastError(); // clear error counter
	tunable.checkLaunchParam(param);
	cudaEventRecord(start, 0);
	for (int i=0; i<tunable.tuningIter(); i++) {
	  if (verbosity >= QUDA_DEBUG_VERBOSE) {
	    printfQuda("About to call tunable.apply block=(%d,%d,%d) grid=(%d,%d,%d) shared_bytes=%d aux=(%d,%d,%d)\n",
		       param.block.x, param.block.y, param.block.z,
		       param.grid.x, param.grid.y, param.grid.z,
		       param.shared_bytes,
		       param.aux.x, param.aux.y, param.aux.z);
	  }
	  tunable.apply(0);  // calls tuneLaunch() again, which simply returns the currently active param
	}
	cudaEventRecord(end, 0);
	cudaEventSynchronize(end);
	cudaEventElapsedTime(&elapsed_time, start, end);
	cudaDeviceSynchronize();
	error = cudaGetLastError();

	{ // check that error state is cleared
	  cudaDeviceSynchronize();
	  cudaError_t error = cudaGetLastError();
	  if (error != cudaSuccess) errorQuda("Failed to clear error state %s\n", cudaGetErrorString(error));
	}

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
	errorQuda("Auto-tuning failed for %s with %s at vol=%s", key.name, key.aux, key.volume);
      }
      if (verbosity >= QUDA_VERBOSE) {
	printfQuda("Tuned %s giving %s for %s with %s\n", tunable.paramString(best_param).c_str(),
		   tunable.perfString(best_time).c_str(), key.name, key.aux);
      }
      time(&now);
      best_param.comment = "# " + tunable.perfString(best_time) + ", tuned ";
      best_param.comment += ctime(&now); // includes a newline

      best_param.time = best_time;

      cudaEventDestroy(start);
      cudaEventDestroy(end);

      if (verbosity >= QUDA_DEBUG_VERBOSE) printfQuda("PostTune %s\n", key.name);
      tunable.postTune();
      param = best_param;
      tunecache[key] = best_param;

    } else if (&tunable != active_tunable) {
      errorQuda("Unexpected call to tuneLaunch() in %s::apply()", typeid(tunable).name());
    }

    // restore the original reduction state
    commGlobalReductionSet(reduceState);

#ifdef PTHREADS
//    pthread_mutex_unlock(&pthread_mutex);
//    tally--;
//    printfQuda("pthread_mutex_unlock b complete %d\n",tally);
#endif

    param.n_calls = profile_count ? 1 : 0;

    return param;
  }

  void printLaunchTimer() {
#ifdef LAUNCH_TIMER
    launchTimer.Print();
#endif
  }
} // namespace quda
