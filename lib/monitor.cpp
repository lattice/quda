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

    std::thread monitor_thread;
    std::atomic<int> check(0);
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;

    static bool initialized = false;

    double energy = 0.0;

    std::list<device::state_t> state_history;

    void device_monitor()
    {
      while (check.load() == 1) {
        auto state = device::get_state();
        state_history.push_back(state);
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }

    void init()
    {
      if (initialized) errorQuda("Monitor thread already initialized");

      start_time = std::chrono::high_resolution_clock::now();

      try { // spawn monitoring thread and release
        monitor_thread = std::thread([&]() { device_monitor(); });
        check.store(1);
      } catch (const std::system_error &e) {
        std::stringstream error;
        error << "Caught system_error with code [" << e.code() << "] meaning [" << e.what() << "]";
        errorQuda("%s", error.str().c_str());
      }
      initialized = true;
    }

    void destroy()
    {
      if (!initialized) errorQuda("Monitor thread not present");
      initialized = false;

      qudaDeviceSynchronize();

      // safely end the monitoring thread
      check.store(0);
      monitor_thread.join(); // thread cleanup

      serialize();
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

  } // namespace monitor
} // namespace quda
