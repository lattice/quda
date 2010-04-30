#ifndef __SHORT_QUDA_H
#define __SHORT_QUDA_H

#define SHORT_LENGTH 65536
#define SCALE_FLOAT ((SHORT_LENGTH-1) * 0.5)
#define SHIFT_FLOAT (-1.f / (SHORT_LENGTH-1))

template <typename Float>
inline short FloatToShort(Float a) {
  //return (short)(a*MAX_SHORT);
  short rtn = (short)((a+SHIFT_FLOAT)*SCALE_FLOAT);
  return rtn;
}

template <typename Float>
inline Float shortToFloat(short a) {
  Float rtn = (float)a/SCALE_FLOAT - SHIFT_FLOAT;
  return rtn;
}



#endif
