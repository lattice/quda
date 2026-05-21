#pragma once

extern "C" {

/**
   @brief Set the default ASAN options.  This ensures that QUDA just
   works when SANITIZE is enabled without requiring ASAN_OPTIONS to
   be set.  We default disable leak checking, otherwise this will
   cause ctest to fail with MPI library leaks.  This declaration
   cannot be in the test library, and must be in the test executable.
*/
const char *__asan_default_options() { return "detect_leaks=0,protect_shadow_gap=0"; }
}
