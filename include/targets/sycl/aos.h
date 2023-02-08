#pragma once

//#include <quda_sycl.h>

namespace quda {

#if 0
  template <typename T> struct subgroup_load_store {
    //using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;
    //using atom_t = std::conditional_t<sizeof(T) % 16 == 0, int4, std::conditional_t<sizeof(T) % 8 == 0, int2, int>>;
    using atom_t = int;
    static_assert(sizeof(T) % 4 == 0, "block_load & block_store do not support sub-word size types");
    static constexpr int n_element = sizeof(T) / sizeof(atom_t);
    using vec = atom_t[n_element];
  };
#endif

  template <typename T, int n> __host__ __device__ void block_load(T out[n], const T *in)
  {
#pragma unroll
    for (int i = 0; i < n; i++) out[i] = in[i];
    //using U = T[n];
    //using LS = subgroup_load_store<U>;
    //using V = typename LS::vec;
    //using A = typename LS::atom_t;
    //constexpr int nv = LS::n_element;
    //auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    //auto vin = reinterpret_cast<const V*>(in) - sg.get_local_id();
    //auto vin = reinterpret_cast<const A*>(in) - nv*sg.get_local_id();
    //auto t = sg.load<nv>(sycl::multi_ptr<const A,sycl::access::address_space::global_space>{vin});
    //#pragma unroll
    //for (int i = 0; i < nv; i++) t[i] = sg.load(vin + sg);
    //auto vout = sg.load(vin);
    //#pragma unroll
    //for (int i = 0; i < n; i++) out[i] = vout[i];
  }

  template <typename T, int n> __host__ __device__ void block_store(T *out, const T in[n])
  {
#pragma unroll
    for (int i = 0; i < n; i++) out[i] = in[i];
  }

  template <typename T> __host__ __device__ void block_load(T &out, const T *in)
  {
    out = *in;
    //auto sg = sycl::ext::oneapi::experimental::this_sub_group();
    //out = sg.load(in - sg.get_local_id());
  }

  template <typename T> __host__ __device__ void block_store(T *out, const T &in)
  {
    *out = in;
  }

}
