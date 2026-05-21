#include <atomic>
#include <thread>
#include <list>
#include <sstream>
#include <fstream>

#include "monitor.h"
#include "device.h"
#include "quda_api.h"
#include "util_quda.h"
#include "tune_quda.h" // hash, version, resource path

namespace quda
{

  namespace monitor
  {

    /**
       Linked list that we record the evolving state of the device being
       monitored
     */
    static std::vector<device::state_t> state_history;

    /**
       @brief Return the time period for the monitor measurements.
       The default is 1000 microsecond, and can be overruled with the
       QUDA_ENABLE_MONITOR_PERIOD environment variable
       @return The time period in microseconds.
     */
    auto get_period()
    {
      static bool init = false;
      static std::chrono::microseconds period = std::chrono::milliseconds(1);
      if (!init) {
        char *period_str = getenv("QUDA_ENABLE_MONITOR_PERIOD");
        if (period_str) period = std::chrono::microseconds(std::atoi(period_str));
        init = true;
      }
      return period;
    }

    /**
       @brief Return if monitoring is enabled.  Default is disabled,
       and can be enabled setting the environment variable
       QUDA_ENABLE_MONITOR=1
     */
    auto is_enabled()
    {
      static bool init = false;
      static bool enable = false;
      if (!init) {
        char *enable_str = getenv("QUDA_ENABLE_MONITOR");
        if (enable_str) {
          if (strcmp(enable_str, "1") == 0) enable = true;
          init = true;
        }
      }
      return enable;
    }

    /**
       Thread that performs the monitoring
     */
    static std::thread monitor_thread;

    /**
       Atomic variable used to signal the monitoring thread
     */
    static std::atomic<bool> is_running(false);

    /**
       @brief The function that is run by the spawned monitor thread
    */
    void device_monitor()
    {
      while (is_running.load()) {
        auto state = device::get_state();
        state_history.push_back(state);

        // periodically reserve larger state size to avoid push_back cost
        if (state_history.size() % 100000 == 0) state_history.reserve(state_history.size() + 100000);

        std::this_thread::sleep_for(get_period());
      }
    }

    /**
       Static variable used to track if we have initiated the
       monitoring
     */
    static bool initialized = false;

    /**
       Static variable used to record the start time
     */
    static std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

    void init()
    {
      if (initialized) errorQuda("Monitor thread already initialized");

      if (is_enabled()) {
        warningQuda("Enabling device monitoring");
        // pre-reserve state_history size to avoid push_back cost
        state_history.reserve(10000);
        start_time = std::chrono::high_resolution_clock::now();

        try { // spawn monitoring thread and release
          monitor_thread = std::thread([&]() { device_monitor(); });
          is_running.store(true);
        } catch (const std::system_error &e) {
          std::stringstream error;
          error << "Caught system_error with code [" << e.code() << "] meaning [" << e.what() << "]";
          errorQuda("%s", error.str().c_str());
        }
        initialized = true;
      }
    }

    void destroy()
    {
      if (initialized) {
        initialized = false;

        qudaDeviceSynchronize();

        // safely end the monitoring thread
        is_running.store(false);
        monitor_thread.join(); // thread cleanup

        serialize();
      }
    }

    void serialize()
    {
      auto resource_path = get_resource_path();
      if (resource_path.empty()) {
        warningQuda("Storing device state disabled");
        return;
      }

      // include current time in filename and rank0 time for all ranks
      std::string serialize_time;
      size_t size;
      if (comm_rank() == 0) {
        auto now_raw = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
        std::stringstream now;
        now << std::put_time(std::localtime(&now_raw), "%Y_%m_%d_%H:%M:%S");
        serialize_time = now.str();
        size = serialize_time.size();
      }
      comm_broadcast(&size, sizeof(size), 0);
      serialize_time.resize(size);
      comm_broadcast(serialize_time.data(), size, 0);

      std::string rank_str = std::to_string(comm_rank());
      std::string monitor_path = resource_path + "/monitor_n" + rank_str + "_" + serialize_time + ".tsv";
      std::ofstream monitor_file;
      monitor_file.open(monitor_path.c_str());
      monitor_file << "monitor"
                   << "\t" << get_quda_version();
      monitor_file << "\t" << get_quda_hash() << std::endl;

      monitor_file << std::setw(12) << "time\t" << std::setw(12) << "power\t" << std::setw(12) << "energy\t";
      monitor_file << std::setw(12) << "temperature\t" << std::setw(12) << "sm-clk\t" << std::endl;

      static uint64_t count = 0;
      static double last_power = 0;
      static double energy = 0.0; // integrated energy
      static std::chrono::time_point<std::chrono::high_resolution_clock> last_time;

      for (auto &state : state_history) {
        std::chrono::duration<float, std::chrono::seconds::period> time = state.time - start_time;
        std::chrono::duration<float, std::chrono::seconds::period> diff = state.time - last_time;

        // trapezoidal integration to compute energy
        if (count > 0) energy += 0.5 * (state.power + last_power) * diff.count();

        monitor_file << std::setw(12) << time.count() << "\t";
        monitor_file << std::setw(12) << state.power << "\t";
        monitor_file << std::setw(12) << energy << "\t";
        monitor_file << std::setw(12) << state.temp << "\t";
        monitor_file << std::setw(12) << state.clock << "\t";
        monitor_file << std::endl;

        last_power = state.power;
        last_time = state.time;
        count++;
      }

      monitor_file.close();
    }

    size_t size() { return is_enabled() ? state_history.size() : 0; }

    state_t mean(size_t start, size_t end)
    {
      state_t mean;
      double last_power = 0.0;
      std::chrono::time_point<std::chrono::high_resolution_clock> last_time;

      if (start > 0 && end > start) {
        auto start_time = state_history[start - 1].time;
        auto end_time = state_history[end - 1].time;
        for (auto i = start; i < end; i++) {
          auto &state = state_history[i];
          if (i - start > 0) {
            std::chrono::duration<float, std::chrono::seconds::period> diff = state.time - last_time;

            // potential for non-uniform samples distribution, so integrate rather than sum
            mean.power += state.power * diff.count();
            mean.temp += state.temp * diff.count();
            mean.clock += state.clock * diff.count();

            // trapezoidal integration to compute energy
            mean.energy += 0.5 * (state.power + last_power) * diff.count();
          }
          last_power = state.power;
          last_time = state.time;
        }

        std::chrono::duration<float, std::chrono::seconds::period> duration = end_time - start_time;
        mean.power /= duration.count();
        mean.temp /= duration.count();
        mean.clock /= duration.count();
      }

      return mean;
    }

  } // namespace monitor
} // namespace quda
