#include <quda_internal.h>
#include <timer.h>
#include <tune_quda.h>

#ifdef INTERFACE_NVTX
#include "nvtx3/nvToolsExt.h"
#endif

namespace quda {

  /**< Print out the profile information */
  void TimeProfile::Print()
  {
    if (profile[QUDA_PROFILE_TOTAL].time > 0.0) {
      printfQuda("\n   %20s Total time = %9.3f secs\n", fname.c_str(), profile[QUDA_PROFILE_TOTAL].time);
    }

    double accounted = 0.0;
    for (int i=0; i<QUDA_PROFILE_COUNT-1; i++) {
      if (profile[i].count > 0) {
        printfQuda("     %20s     = %9.3f secs (%7.3f%%),\t with %8d calls at %6.3e us per call\n",
                   (const char *)&pname[i][0], profile[i].time, 100 * profile[i].time / profile[QUDA_PROFILE_TOTAL].time,
                   profile[i].count, 1e6 * profile[i].time / profile[i].count);
        accounted += profile[i].time;
      }
    }
    if (accounted > 0.0) {
      double missing = profile[QUDA_PROFILE_TOTAL].time - accounted;
      printfQuda("        total accounted       = %9.3f secs (%7.3f%%)\n", accounted,
                 100 * accounted / profile[QUDA_PROFILE_TOTAL].time);
      printfQuda("        total missing         = %9.3f secs (%7.3f%%)\n", missing,
                 100 * missing / profile[QUDA_PROFILE_TOTAL].time);
    }

    if (accounted > profile[QUDA_PROFILE_TOTAL].time) {
      warningQuda("Accounted time %9.3f secs in %s is greater than total time %9.3f secs", accounted,
                  (const char *)&fname[0], profile[QUDA_PROFILE_TOTAL].time);
    }
  }

  std::string TimeProfile::pname[] = {"download",
                                      "upload",
                                      "init",
                                      "preamble",
                                      "compute",
                                      "training",
                                      "comms",
                                      "epilogue",
                                      "free",
                                      "file i/o",
                                      "chronology",
                                      "eigen",
                                      "eigenLU",
                                      "eigenEV",
                                      "eigenQR",
                                      "arpack",
                                      "host compute",
                                      "dummy",
                                      "pack kernel",
                                      "dslash kernel",
                                      "gather",
                                      "scatter",
                                      "kernel launch",
                                      "event record",
                                      "event query",
                                      "stream wait event",
                                      "set func attribute",
                                      "event synchronize",
                                      "stream synchronize",
                                      "device synchronize",
                                      "memcpy d2d async",
                                      "memcpy d2h async",
                                      "memcpy2d d2h async",
                                      "memcpy h2d async",
                                      "memcpy default async",
                                      "comms start",
                                      "comms query",
                                      "constant",
                                      "total"};

#ifdef INTERFACE_NVTX
  const uint32_t TimeProfile::nvtx_colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
  const int TimeProfile::nvtx_num_colors = sizeof(nvtx_colors)/sizeof(uint32_t);
#endif

  // global timer
  host_timer_t global_profile[QUDA_PROFILE_COUNT] = {};
  static bool global_switchOff[QUDA_PROFILE_COUNT] = {};
  static int global_total_level[QUDA_PROFILE_COUNT] = {};

  void TimeProfile::StopGlobal(const char *func, const char *file, int line, QudaProfileType idx)
  {
    global_total_level[idx]--;
    if (global_total_level[idx] == 0) global_profile[idx].stop(func, file, line);

    // switch off total timer if we need to
    if (global_switchOff[idx]) {
      global_total_level[idx]--;
      if (global_total_level[idx] == 0) global_profile[idx].stop(func, file, line);
      global_switchOff[idx] = false;
    }
  }

  void TimeProfile::StartGlobal(const char *func, const char *file, int line, QudaProfileType idx)
  {
    // if total timer isn't running, then start it running
    if (!global_profile[idx].running) {
      global_profile[idx].start(func, file, line);
      global_total_level[idx]++;
      global_switchOff[idx] = true;
    }

    if (global_total_level[idx] == 0) global_profile[idx].start(func, file, line);
    global_total_level[idx]++;
  }

#ifdef INTERFACE_NVTX

#define PUSH_RANGE(name, cid)                                                                                          \
  {                                                                                                                    \
    int color_id = cid;                                                                                                \
    color_id = color_id % nvtx_num_colors;                                                                             \
    nvtxEventAttributes_t eventAttrib = {};                                                                            \
    eventAttrib.version = NVTX_VERSION;                                                                                \
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                                                                  \
    eventAttrib.colorType = NVTX_COLOR_ARGB;                                                                           \
    eventAttrib.color = nvtx_colors[color_id];                                                                         \
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;                                                                 \
    eventAttrib.message.ascii = name;                                                                                  \
    eventAttrib.category = cid;                                                                                        \
    nvtxRangePushEx(&eventAttrib);                                                                                     \
  }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

  void TimeProfile::StartTotal(const char *func, const char *file, int line, QudaProfileType idx)
  {
    // if total timer isn't running, then start it running
    if (!profile[QUDA_PROFILE_TOTAL].running && idx != QUDA_PROFILE_TOTAL) {
      profile[QUDA_PROFILE_TOTAL].start(func, file, line);
      switchOff = true;
    }
  }

  void TimeProfile::StopTotal(const char *func, const char *file, int line, QudaProfileType idx)
  {
    // switch off total timer if we need to
    if (switchOff && idx != QUDA_PROFILE_TOTAL) {
      profile[QUDA_PROFILE_TOTAL].stop(func, file, line);
      switchOff = false;
    }
  }

  void TimeProfile::Start_(const char *func, const char *file, int line, QudaProfileType idx)
  {
    StartTotal(func, file, line, idx);

    // if a timer is already running, stop it and push to stack
    for (auto i = 0; i < QUDA_PROFILE_COUNT - 1; i++) {
      if (profile[i].running) {
        if ((i == QUDA_PROFILE_COMPUTE || i == QUDA_PROFILE_H2D || i == QUDA_PROFILE_D2H)
            && i != idx) // don't synchronize if nesting the same profile type
          qudaDeviceSynchronize();
        profile[i].stop(file, func, line);
        if (use_global) StopGlobal(func, file, line, static_cast<QudaProfileType>(i));
        POP_RANGE;
        pt_stack.push(static_cast<QudaProfileType>(i));
      }
    }

    profile[idx].start(func, file, line);
    PUSH_RANGE(fname.c_str(), idx);
    if (use_global) StartGlobal(func, file, line, idx);
  }

  void TimeProfile::Stop_(const char *func, const char *file, int line, QudaProfileType idx)
  {
    auto i = !pt_stack.empty() ? pt_stack.top() : QUDA_PROFILE_COUNT;
    if ((idx == QUDA_PROFILE_COMPUTE || idx == QUDA_PROFILE_H2D || idx == QUDA_PROFILE_D2H)
        && i != idx)           // don't synchronize if nesting same profile type
      qudaDeviceSynchronize(); // ensure accurate profiling

    if (!profile[idx].stop(func, file, line)) {
      for (auto i = 0; i < QUDA_PROFILE_COUNT - 1; i++)
        if (profile[i].running) errorQuda("Failed to stop timer idx = %d, however idx = %d is running", idx, i);
      errorQuda("Failed to stop timer idx = %d", idx);
    }
    if (use_global) StopGlobal(func, file, line, idx);
    POP_RANGE;

    if (pt_stack.empty()) {
      StopTotal(func, file, line, idx);
    } else {
      // restore any pre-existing timers if needed
      auto i = pt_stack.top();
      pt_stack.pop();
      profile[i].start(func, file, line);
      PUSH_RANGE(fname.c_str(), i);
      if (use_global) StartGlobal(func, file, line, i);
    }
  }

#undef PUSH_RANGE
#undef POP_RANGE

  void TimeProfile::PrintGlobal() {
    if (global_profile[QUDA_PROFILE_TOTAL].time > 0.0) {
      printfQuda("\n   %20s Total time = %9.3f secs\n", "QUDA", global_profile[QUDA_PROFILE_TOTAL].time);
    }

    double accounted = 0.0;
    bool print_timer = true; // whether to print that timer
    for (int i=0; i<QUDA_PROFILE_LOWER_LEVEL; i++) { // we do not want to print detailed lower level timers
      if (global_profile[i].count > 0) {
        if (print_timer)
          printfQuda("     %20s     = %9.3f secs (%7.3f%%),\t with %8d calls at %6.3e us per call\n",
                     (const char *)&pname[i][0], global_profile[i].time,
                     100 * global_profile[i].time / global_profile[QUDA_PROFILE_TOTAL].time, global_profile[i].count,
                     1e6 * global_profile[i].time / global_profile[i].count);
        accounted += global_profile[i].time;
      }
    }
    if (accounted > 0.0) {
      double missing = global_profile[QUDA_PROFILE_TOTAL].time - accounted;
      printfQuda("        total accounted       = %9.3f secs (%7.3f%%)\n", accounted,
                 100 * accounted / global_profile[QUDA_PROFILE_TOTAL].time);
      printfQuda("        total missing         = %9.3f secs (%7.3f%%)\n", missing,
                 100 * missing / global_profile[QUDA_PROFILE_TOTAL].time);
    }

    if (accounted > global_profile[QUDA_PROFILE_TOTAL].time) {
      warningQuda("Accounted time %9.3f secs in %s is greater than total time %9.3f secs\n", accounted, "QUDA",
                  global_profile[QUDA_PROFILE_TOTAL].time);
    }
  }

  TimeProfile dummy("default", false);

  static std::stack<TimeProfile *> tp_stack;

  pushProfile::pushProfile(TimeProfile &profile, double &secs, double &gflops) :
    profile(profile), secs(secs), gflops(gflops), flops(Tunable::flops_global())

  {
    if (profile.Name() != getProfile().Name()) {
      // only push to stack if this profile not already the active one
      profile.TPSTART(QUDA_PROFILE_TOTAL);
      tp_stack.push(&profile);
      active = true;
    }
  }

  pushProfile::~pushProfile()
  {
    if (active == true) {
      if (tp_stack.empty()) errorQuda("popProfile() called with empty stack");
      auto &profile = *(tp_stack.top());
      if (&(this->profile) != &profile) errorQuda("Popped profile is not the expected one");
      tp_stack.pop();
      profile.TPSTOP(QUDA_PROFILE_TOTAL);
      secs = profile.Last(QUDA_PROFILE_TOTAL);
      gflops = (Tunable::flops_global() - flops) * 1e-9;
      if (&gflops != &gflops_dummy) comm_allreduce_sum(gflops);
    }
  }

  TimeProfile &getProfile()
  {
    if (tp_stack.empty()) return dummy;
    return *(tp_stack.top());
  }
}
