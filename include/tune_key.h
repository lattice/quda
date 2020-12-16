#pragma once

#include <cstring>

namespace quda {

  struct TuneKey {

    static const int volume_n = 32;
    static const int name_n = 512;
    static const int aux_n = 256;
    char volume[volume_n];
    char name[name_n];
    char aux[aux_n];

    TuneKey() { }
    TuneKey(const char v[], const char n[], const char a[]="type=default") {
      strcpy(volume, v);
      strcpy(name, n);
      strcpy(aux, a);
    } 

    TuneKey(const TuneKey &) = default;
    TuneKey(TuneKey &&) = default;
    TuneKey& operator=(const TuneKey &) = default;
    TuneKey& operator=(TuneKey &&) = default;

    bool operator<(const TuneKey &other) const {
      int vc = std::strcmp(volume, other.volume);
      if (vc < 0) {
	return true;
      } else if (vc == 0) {
	int nc = std::strcmp(name, other.name);
	if (nc < 0) {
	  return true;
	} else if (nc == 0) {
	  return (std::strcmp(aux, other.aux) < 0 ? true : false);
	}
      }
      return false;
    }
  
  };

  /** Return the key of the last kernel that has been tuned / called.*/
  TuneKey getLastTuneKey();

}
