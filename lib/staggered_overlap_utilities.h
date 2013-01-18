#ifndef STAGGERED_OVERLAP_UTILITIES_H
#define STAGGERED_OVERLAP_UTILITIES_H


template<int ghostDir, int Nface>
struct NeighborIndex{

  // returns -1 if neighbor is out of bounds
  // use this to determine whether site is updated
  template<KernelType Dir>
  __host__ __device__ static int forward(int x1, int x2, int x3, int x4, const DslashParam& param){ return Dir; }

  template<KernelType Dir>
  __host__ __device__ static int back(int x1, int x2, int x3, int x4, const DslashParam& param){ return Dir; }
};

template<int Nface>
struct NeighborIndex<0, Nface>
{

  template<KernelType Dir>
  __device__ static int forward(int x1, int x2, int x3, int x4, const DslashParam& param){

    if(Dir==0 && x1 == Nface-1) return (x4*X3X2X1 + x3*X2X1 + x2*X1) >> 1;

    int dirOffset=0;
    if(x1 >= Nface) x1 -= Nface; dirOffset = 3*Nface*(X4X3X2>>1);

    switch(Dir){
      case 0:
	if(x1 == Nface-1) return -1;
        x1 += 1;
      break;

      case 1: 
	if(x2 == X2m1) return -1;
        x2 += 1; 
      break;

      case 2: 
	if(x3 == X3m1) return -1;
        x3 += 1; 
      break;

      case 3: 
        if(x4 == X4m1) return -1;
        x4 += 1; 
      break;
    }
    return param.ghostOffset[0] + dirOffset + ((x1*X4X3X2 + x4*X3X2 + x3*X2 + x2) >> 1);
  } 



  template<KernelType Dir> 
  __device__ static int forward_three(int x1, int x2, int x3, int x4, const DslashParam& param){

    if(Dir == 0 && x1 < Nface && x1 >= Nface-3) return (x4*X3X2X1 + x3*X2X1 + x2*X1 + x1-Nface + 3)>>1; 

    int dirOffset=0;
    if(x1 >= Nface) x1 -= Nface; dirOffset = 3*Nface*(X4X3X2>>1); 
    
    switch(Dir){
      case 0: 
        if(x1 >= Nface-3) return -1;
	x1 += 3;
      break;

      case 1:
	if(x2 >= X2m3) return -1;
        x2 += 3;
      break;

      case 2:
	if(x3 >= X3m3) return -1;
	x3 += 3;
      break;

      case 3:
	if(x4 >= X4m3) return -1;
	x4 += 3;
      break;
    }   
    return param.ghostOffset[0] + dirOffset + ((x1*X3X2X1 + x4*X3X2 + x3*X2 + x2)>>1);
  }



  template<KernelType Dir> 
  __device__ static int back(int x1, int x2, int x3, int x4, const DslashParam& param){
    

    if(Dir == 0 && x1 == Nface) return (x4*X3X2X1 + x3*X2X1 + x2*X1 + X1m1) >> 1;
    

    int dirOffset = 0;
    if(x1 >= Nface) x1 -= Nface; dirOffset = 3*Nface*(X4X3X2>>1);

    switch(Dir){
      case 0:
        if(x1 == 0) return -1;
        x1 -= 1;
      break;

      case 1:
	if(x2 == 0) return -1;
        x2 -= 1;
      break;

      case 2:
        if(x3 == 0) return -1;
        x3 -= 1;
      break;

      case 3:
        if(x4 == 0) return -1;
        x4 -= 1;
      break;
    }
    return param.ghostOffset[0] + dirOffset + ((x1*X4X3X2 + x4*X3X2 + x3*X2 + x2) >> 1);
  }


  template<KernelType Dir> 
  __device__ static int back_three(int x1, int x2, int x3, int x4, const DslashParam& param){

    if(Dir == 0 && x1 >= Nface && x1 < Nface+3) return (x4*X3X2X1 + x3*X2X1 + x2*X1 + X1m3 + x1-Nface)>>1; 

    int dirOffset=0;
    if(x1 >= Nface) x1 -= Nface; dirOffset = 3*Nface*(X4X3X2>>1);    
 
    switch(Dir){
      case 0: 
        if(x1 < 3) return -1;
	x1 -= 3;
      break;

      case 1:
	if(x2 < 3) return -1;
        x2 -= 3;
      break;

      case 2:
	if(x3 < 3) return -1;
	x3 -= 3;
      break;

      case 3:
	if(x4 < 3) return -1;
	x4 -= 3;
      break;
    }   
    return param.ghostOffset[0] + dirOffset + ((x1*X3X2X1 + x4*X3X2 + x3*X2 + x2)>>1);
  }
};

/*
template<KernelType Dir>
void getCoordinates(int* x1_p, int* x2_p, int* x3_p, int* x4_p, 
		    int cb_index, int parity){ return; }


template<>
void getCoordinates<0>(int* const x1_p, int* const x2_p, 
		       int* const x3_p, int* const x4_p,
		       int cb_index, int parity)
{
  // cb_idx = (x1*X4X3X2 + x4*X3X2 + x3*X2 + x2)/2
  const int x2h = cb_index % X2h;
  *x3_p = (cb_index/X2h) % X3;
  *x1_p = cb_index/X4X3X2h;
  *x4_p = (cb_index/(X3X2>>1)) % X4;
  const int x2odd = (*x1_p + *x3_p + *x2_p + parity) & 1;
  *x2_p = 2*x2h + x2odd;
  return;
}


template<>
void getCoordinates<1>(int* const x1_p, int* const x2_p, 
		       int* const x3_p, int* const x4_p,
		       int cb_index, int parity)
{
  // cb_index = (x2*X4X3X1 + x4*X3X1 + x3*X1 + x1)/2
  const int x1h = cb_index % X1h;
  *x3_p = (cb_index/X1h) % X3;
  *x2_p = cb_index/X4X3X1h;
  *x4_p = (cb_index/(X3X1>>1)) % X4;
  const int x1odd = (*x2_p + *x3_p + *x4_p + parity) & 1;
  *x1_p = 2*x1h + x1odd;

  return;
}

template<>
void getCoordinates<2>(int* const x1_p, int* const x2_p,
		       int* const x3_p, int* const x4_p,
		       int cb_index, int parity)
{
  // cb_index = (x3*X4X2X1 + x4*X2X1 + x2*X1 + x1)/2
  const int x1h = cb_index % X1h;
  *x2_p = (cb_index/X1h) % X2;
  *x3_p = cb_index/X4X2X1h;
  *x4_p = (cb_index/(X2X1>>1)) % X4;
  const int x1odd = (*x2_p + *x3_p + *x4_p + parity) & 1;
  *x1_p = 2*x1h + x1odd;

  return;
}

template<>
void getCoordinates<3>(int* const x1_p, int* const x2_p,
		       int* const x3_p, int* const x4_p,
		       int cb_index, int parity)
{
  // cb_index = (x4*X3X2X1 + x3*X2X1 + x2*X1 + x1)/2
  // Note that this is the canonical ordering
  const int x1h = cb_index % X1h;
  *x2_p = (cb_index/X1h) % X2;
  *x4_p = (cb_index/X3X2X1h);
  *x3_p = (cb_index/(X2X1>>1)) % X3;
  const int x1odd = (*x2_p + *x3_p + *x4_p + parity) & 1;
  *x1_p = 2*x1h + x1odd;

  return;
}

*/

template<KernelType Dir> 
__device__ void getCoordinates(int* const x1_p, int* const x2_p,
		    int* const x3_p, int* const x4_p,
		    int cb_index, int parity)
{

  int xh, xodd;
  switch(Dir){
    case 0:
      // cb_idx = (x1*X4X3X2 + x4*X3X2 + x3*X2 + x2)/2
      xh = cb_index % X2h;
      *x3_p = (cb_index/X2h) % X3;
      *x1_p = cb_index/X4X3X2h;
      *x4_p = (cb_index/(X3X2>>1)) % X4;
      xodd = (*x1_p + *x3_p + *x2_p + parity) & 1;
      *x2_p = 2*xh + xodd;
    break;

    case 1:
      // cb_index = (x2*X4X3X1 + x4*X3X1 + x3*X1 + x1)/2
      xh = cb_index % X1h;
      *x3_p = (cb_index/X1h) % X3;
      *x2_p = cb_index/X4X3X1h;
      *x4_p = (cb_index/(X3X1>>1)) % X4;
      xodd = (*x2_p + *x3_p + *x4_p + parity) & 1;
      *x1_p = 2*xh + xodd;
    break;
    
    case 2:
      // cb_index = (x3*X4X2X1 + x4*X2X1 + x2*X1 + x1)/2
      xh = cb_index % X1h;
      *x2_p = (cb_index/X1h) % X2;
      *x3_p = cb_index/X4X2X1h;
      *x4_p = (cb_index/(X2X1>>1)) % X4;
      xodd = (*x2_p + *x3_p + *x4_p + parity) & 1;
      *x1_p = 2*xh + xodd;
    break; 

    case 3:
     // cb_index = (x4*X3X2X1 + x3*X2X1 + x2*X1 + x1)/2
     // Note that this is the canonical ordering in the interior region.
     xh = cb_index % X1h;
     *x2_p = (cb_index/X1h) % X2;
     *x4_p = (cb_index/(X3X2X1>>1));
     *x3_p = (cb_index/(X2X1>>1)) % X3;
     xodd = (*x2_p + *x3_p + *x4_p + parity) & 1;
     *x1_p = 2*xh + xodd;
    break;

    default:
    break;
  } // switch(Dir)

  return;
}

template<KernelType Dir, int Nface>
__device__ void getGluonCoordsFromGhostCoords(int* const y1_p, int* const y2_p, int* const y3_p, int* const y4_p,
				   int x1, int x2, int x3, int x4)
{

  *y1_p = x1;
  *y2_p = x2; 
  *y3_p = x3;
  *y4_p = x4;

  switch(Dir){
    case 0:
      if(x1 >= Nface) *y1_p += X1;
      if(Y2 > X2) *y2_p += Nface;
      if(Y3 > X3) *y3_p += Nface;
      if(Y4 > X4) *y4_p += Nface;
    break;

    case 1:
      if(x2 >= Nface) *y2_p += X2;
      if(Y1 > X1) *y1_p += Nface;
      if(Y3 > X3) *y3_p += Nface;
      if(Y4 > X4) *y4_p += Nface;
    break;

    case 2:
      if(x3 >= Nface) *y3_p += X3;
      if(Y1 > X1) *y1_p += Nface;
      if(Y2 > X2) *y2_p += Nface;
      if(Y4 > X4) *y4_p += Nface;
    break;

    case 3:
      if(x4 >= Nface) *y4_p += X4;
      if(Y1 > X1) *y1_p += Nface;
      if(Y2 > X2) *y2_p += Nface;
      if(Y3 > X3) *y3_p += Nface;
    break;

    default:
    break;
  }
  return;
}


template<KernelType Dir, int Nface>
__device__ int getGluonFullIndexFromGhostIndex(int x1, int x2, int x3, int x4, int parity)
{
  // Y4, Y3, Y2, Y1 are the dimensions of the extended domain
  switch(Dir){
    case 0:
      if(x1 >= Nface) x1 += X1;
      // shift so that the sites are within the Red Cross
      if(Y2 > X2) x2 += Nface;
      if(Y3 > X3) x3 += Nface;
      if(Y4 > X4) x4 += Nface;
    break;

    case 1:
      if(x2 >= Nface) x2 += X2;
      if(Y1 > X1) x1 += Nface;
      if(Y3 > X3) x3 += Nface;
      if(Y4 > X4) x4 += Nface;
    break;

    case 2:
      if(x3 >= Nface) x3 += X3;
      if(Y1 > X1) x1 += Nface;
      if(Y2 > X2) x2 += Nface;
      if(Y4 > X4) x4 += Nface;
    break;

    case 3:
      if(x4 >= Nface) x4 += X4;
      if(Y1 > X1) x1 += Nface;
      if(Y2 > X2) x2 += Nface;
      if(Y3 > X3) x3 += Nface;
    break;

    default:
    break;
  } // switch(Dir)

  return  x4*Y3Y2Y1 + x3*Y2Y1 + x2*Y1 + x1;
  // note that I haven't checked the parity. 
  // I need to do this if I want to allow overlaps with an odd number of sites 
}


// get the "checker-board" index
template<KernelType Dir, int Nface>
__device__ int getGluonCBIndexFromGhostIndex(int x1, int x2, int x3, int x4, int parity)
{
  return getGluonFullIndexFromGhostIndex<Dir,Nface>(x1, x2, x3, x4, parity) >> 1;
}



template<typename T>
void MatMulVectAppend(T (&vout)[3], const T(& mat)[9], const T (&vin)[3])
{
  // real part
  vout[0].x += mat[0].x*vin[0].x - mat[0].y*vin[0].y;
  vout[0].x += mat[1].x*vin[1].x - mat[1].y*vin[1].y;
  vout[0].x += mat[2].x*vin[2].x - mat[2].y*vin[2].y;

  vout[1].x += mat[3].x*vin[0].x - mat[3].y*vin[0].y;
  vout[1].x += mat[4].x*vin[1].x - mat[4].y*vin[1].y;
  vout[1].x += mat[5].x*vin[2].x - mat[5].y*vin[2].y;

  vout[2].x += mat[6].x*vin[0].x - mat[6].y*vin[0].y;
  vout[2].x += mat[7].x*vin[1].x - mat[7].y*vin[1].y;
  vout[2].x += mat[8].x*vin[2].x - mat[8].y*vin[2].y;

  // imaginary part
  vout[0].y += mat[0].x*vin[0].y + mat[0].y*vin[0].x;
  vout[0].y += mat[1].x*vin[1].y + mat[1].y*vin[1].x;
  vout[0].y += mat[2].x*vin[2].y + mat[2].y*vin[2].x;

  vout[1].y += mat[0].x*vin[0].y + mat[0].y*vin[0].x;
  vout[1].y += mat[1].x*vin[1].y + mat[1].y*vin[1].x;
  vout[1].y += mat[2].x*vin[2].y + mat[2].y*vin[2].x;

  vout[2].y += mat[6].x*vin[0].y + mat[6].y*vin[0].x;
  vout[2].y += mat[7].x*vin[1].y + mat[7].y*vin[1].x;
  vout[2].y += mat[8].x*vin[2].y + mat[8].y*vin[2].x;

  return;
}


void MatMulVecAppend(float2 (&vout)[3], const float4(& mat)[5], const float2 (&vin)[3])
{

  // real part 
  vout[0].x += mat[0].x*vin[0].x - mat[0].y*vin[0].y;
  vout[0].x += mat[0].z*vin[1].x - mat[0].w*vin[1].y;
  vout[0].x += mat[1].x*vin[2].x - mat[1].y*vin[2].y;

  vout[1].x += mat[1].z*vin[0].x - mat[3].w*vin[0].y;
  vout[1].x += mat[2].x*vin[1].x - mat[2].y*vin[1].y;
  vout[1].x += mat[2].z*vin[2].x - mat[2].w*vin[2].y;

  vout[2].x += mat[3].x*vin[0].x - mat[3].y*vin[0].y;
  vout[2].x += mat[3].z*vin[1].x - mat[3].w*vin[1].y;
  vout[2].x += mat[4].x*vin[2].x - mat[4].y*vin[2].y;

  // imaginary part
  vout[0].y += mat[0].x*vin[0].y + mat[0].y*vin[0].x;
  vout[0].y += mat[0].z*vin[1].y + mat[0].w*vin[1].x;
  vout[0].y += mat[1].x*vin[2].y + mat[1].y*vin[2].x;

  vout[1].y += mat[1].z*vin[0].y + mat[3].w*vin[0].x;
  vout[1].y += mat[2].x*vin[1].y + mat[2].y*vin[1].x;
  vout[1].y += mat[2].z*vin[2].y + mat[2].w*vin[2].x;

  vout[2].y += mat[3].x*vin[0].y + mat[3].y*vin[0].x;
  vout[2].y += mat[3].z*vin[1].y + mat[3].w*vin[1].x;
  vout[2].y += mat[4].x*vin[2].y + mat[4].y*vin[2].x;

  return;
}



// template the kernel so that 
// we can switch between single-/double- and half- precision

template<KernelType Dir, int Nface>
__device__ int getSpinorStride(int cb_index)
{
  const int stride = (cb_index < Vh) ? sp_stride : Nface*ghostFace[Dir];
  return stride;
}


template<KernelType Dir, int Nface>
__device__ int getNormIndex(int cb_index, const DslashParam& param)
{
  if(cb_index < Vh) return cb_index;

  const int offset = param.ghostOffset[Dir] + 3*Nface*ghostFace[Dir];
  const int norm_idx = (cb_index >= offset) ? cb_index - offset + param.ghostNormOffset[Dir] + Nface*ghostFace[Dir] : cb_index - param.ghostOffset[Dir] + param.ghostNormOffset[Dir];

  return norm_idx;
}




#endif
