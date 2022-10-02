#pragma once

#include <string>

namespace quda {

  /**
     @brief query if tuning is in progress
     @return tuning in progress?
  */
  bool activeTuning();

  void loadTuneCache();
  void saveTuneCache(bool error = false);

  /**
   * @brief Save profile to disk.
   */
  void saveProfile(const std::string label = "");

  /**
   * @brief Flush profile contents, setting all counts to zero.
   */
  void flushProfile();

  /**
   * @brief Post an event in the trace, recording where it was posted
   */
  void postTrace_(const char *func, const char *file, int line);

  /**
   * @brief Enable the profile kernel counting
   */
  void enableProfileCount();

  /**
   * @brief Disable the profile kernel counting
   */
  void disableProfileCount();

  /**
   * @brief Enable / disable whether are tuning a policy
   */
  void setPolicyTuning(bool);

  /**
   * @brief Query whether we are currently tuning a policy
   */
  bool policyTuning();

  /**
   * @brief Enable / disable whether we are tuning an uber kernel
   */
  void setUberTuning(bool);

  /**
   * @brief Query whether we are tuning an uber kernel
   */
  bool uberTuning();

} // namespace quda

#define postTrace() quda::postTrace_(__func__, quda::file_name(__FILE__), __LINE__)
