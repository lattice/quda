#if (__COMPUTE_CAPACITY__ < 130)

__device__ inline void dsadd(volatile float2 &c, const volatile float2 &a, const volatile float2 &b) {
  float t1 = a.x + b.x;
  float e = t1 - a.x;
  float t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;
  // The result is t1 + t2, after normalization.
  c.x = e = t1 + t2;
  c.y = t2 - (e - t1);
}

struct doublesingle {
  float2 a;
  __device__ inline doublesingle() : a(make_float2(0.0f,0.0f)) { ; } 
  __device__ inline doublesingle(const volatile doublesingle &b) : a(make_float2(b.a.x, b.a.y)) { ; } 
  __device__ inline doublesingle(const float a) : a(make_float2(a, 0.0)) { ; }

  __device__ inline void operator+=(const doublesingle &b) { dsadd(this->a, this->a, b.a); }
  __device__ inline void operator+=(const float &b) { 
    float2 b2 = make_float2(b, 0.0); 
    dsadd(this->a, this->a, b2); }
  
  __device__ inline doublesingle& operator=(const doublesingle &b)
    { a.x = b.a.x; a.y = b.a.y; return *this; }
    
  __device__ inline doublesingle& operator=(const float &b) 
  { a.x = b; a.y = 0.0f; return *this; }
};

__device__ inline volatile doublesingle operator+=(volatile doublesingle& a, const volatile doublesingle &b) 
{ dsadd(a.a, a.a, b.a); return a; }
__host__ double operator+=(double& a, doublesingle &b) { a = b.a.x; a += b.a.y; return a; }

struct doublesingle2 { 
  doublesingle x; 
  doublesingle y; 
  __device__ inline doublesingle2& operator=(const double &a) 
  { x = a; y = a; return *this; }
};
struct doublesingle3 { doublesingle x; doublesingle y; doublesingle z; };

#endif
