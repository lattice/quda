#ifndef _TUNE_KEY_H
#define _TUNE_KEY_H

#include <cstring>

namespace quda {

  struct TuneKey {

    char volume[32];
    char name[256];
    char aux[256];

    TuneKey() { }
    TuneKey(const char v[], const char n[], const char a[]="type=default") {
      strcpy(volume, v);
      strcpy(name, n);
      strcpy(aux, a);
    } 
    TuneKey(const TuneKey &key) {
      strcpy(volume,key.volume);
      strcpy(name,key.name);
      strcpy(aux,key.aux);
    }

    TuneKey& operator=(const TuneKey &key) {
      if (&key != this) {
	strcpy(volume,key.volume);
	strcpy(name,key.name);
	strcpy(aux,key.aux);
      }
      return *this;
    }

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

}

/** Return the key of the last kernel that has been tuned / called.*/
quda::TuneKey getLastTuneKey();
 
#endif
