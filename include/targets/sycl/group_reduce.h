#pragma once

#include <quda_sycl_api.h>
#include <array.h>
using quda::array;

template <typename T>
inline void blockReduceSum(sycl::group<3> grp, T &out, const T &in)
{
  out = sycl::reduce_over_group(grp, in, sycl::plus<>());
}

template <typename T, int N>
inline std::enable_if_t<(N==1)||(N==2)||(N==3)||(N==4)||(N==8)||(N==16),void>
blockReduceSum(sycl::group<3> grp, array<T,N> &out, const array<T,N> &in)
{
  auto inx = reinterpret_cast<const sycl::vec<T,N>*>(&in);
  auto outx = sycl::reduce_over_group(grp, *inx, sycl::plus<>());
  out = *reinterpret_cast<array<T,N>*>(&outx);
}

template <typename T, int N>
inline std::enable_if_t<N==6,void>
blockReduceSum(sycl::group<3> grp, array<T,N> &out, const array<T,N> &in)
{
  for(int i=0; i<2; i++) {
    auto inx = reinterpret_cast<const array<T,3>*>(&in[3*i]);
    auto outx = reinterpret_cast<array<T,3>*>(&out[3*i]);
    blockReduceSum(grp, *outx, *inx);
  }
}

template <typename T, int N>
inline std::enable_if_t<N==32,void>
blockReduceSum(sycl::group<3> grp, array<T,N> &out, const array<T,N> &in)
{
  for(int i=0; i<2; i++) {
    auto inx = reinterpret_cast<const array<T,16>*>(&in[16*i]);
    auto outx = reinterpret_cast<array<T,16>*>(&out[16*i]);
    blockReduceSum(grp, *outx, *inx);
  }
}

template <typename T, int M, int N>
inline void blockReduceSum(sycl::group<3> grp, array<array<T,M>,N> &out,
			   const array<array<T,M>,N> &in)
{
  const int N2 = M * N;
  auto inx = reinterpret_cast<const array<T,N2>*>(&in);
  auto outx = reinterpret_cast<array<T,N2>*>(&out);
  blockReduceSum(grp, *outx, *inx);
}

template <typename T, int N>
inline void blockReduceSum(sycl::group<3> grp, array<vec2<T>,N> &out,
			   const array<vec2<T>,N> &in)
{
  const int N2 = 2 * N;
  auto inx = reinterpret_cast<const array<T,N2>*>(&in);
  auto outx = reinterpret_cast<array<T,N2>*>(&out);
  blockReduceSum(grp, *outx, *inx);
}

template <typename T>
inline void blockReduceSum(sycl::group<3> grp, quda::complex<T> &out,
			   const quda::complex<T> &in)
{
  auto inx = reinterpret_cast<const array<T,2>*>(&in);
  auto outx = reinterpret_cast<array<T,2>*>(&out);
  blockReduceSum(grp, *outx, *inx);
}

template <typename T, int N>
inline void blockReduceSum(sycl::group<3> grp, array<quda::complex<T>,N> &out,
			   const array<quda::complex<T>,N> &in)
{
  const int N2 = 2 * N;
  auto inx = reinterpret_cast<const array<T,N2>*>(&in);
  auto outx = reinterpret_cast<array<T,N2>*>(&out);
  blockReduceSum(grp, *outx, *inx);
}

template <typename T>
inline void blockReduceSum(sycl::group<3> grp, vec2<T> &out, const vec2<T> &in)
{
  auto inx = reinterpret_cast<const array<T,2>*>(&in);
  auto outx = reinterpret_cast<array<T,2>*>(&out);
  blockReduceSum(grp, *outx, *inx);
}

template <typename T>
inline void blockReduceSum(sycl::group<3> grp, vec3<T> &out, const vec3<T> &in)
{
  auto inx = reinterpret_cast<const array<T,3>*>(&in);
  auto outx = reinterpret_cast<array<T,3>*>(&out);
  blockReduceSum(grp, *outx, *inx);
}

template <typename T>
inline void blockReduceSum(sycl::group<3> grp, vec4<T> &out, const vec4<T> &in)
{
  auto inx = reinterpret_cast<const array<T,4>*>(&in);
  auto outx = reinterpret_cast<array<T,4>*>(&out);
  blockReduceSum(grp, *outx, *inx);
}


template <typename T>
inline void blockReduceMax(sycl::group<3> grp, T &out, const T &in)
{
  out = sycl::reduce_over_group(grp, in, sycl::maximum<>());
}

template <typename T, int N>
inline std::enable_if_t<(N==1)||(N==2)||(N==3)||(N==4)||(N==8)||(N==16),void>
blockReduceMax(sycl::group<3> grp, array<T,N> &out, const array<T,N> &in)
{
  auto inx = reinterpret_cast<const sycl::vec<T,N>*>(&in);
  auto outx = sycl::reduce_over_group(grp, *inx, sycl::maximum<>());
  out = *reinterpret_cast<array<T,N>*>(&outx);
}

template <typename T>
inline void blockReduceMax(sycl::group<3> grp, quda::deviation_t<T> &out, const quda::deviation_t<T> &in)
{
  auto inx = reinterpret_cast<const sycl::vec<T,2>*>(&in);
  auto outx = sycl::reduce_over_group(grp, *inx, sycl::maximum<>());
  out = *reinterpret_cast<quda::deviation_t<T>*>(&outx);
}


template <typename T>
inline void blockReduceMin(sycl::group<3> grp, T &out, const T &in)
{
  out = sycl::reduce_over_group(grp, in, sycl::minimum<>());
}


template <typename T, typename R>
inline std::enable_if_t<std::is_same_v<typename R::reducer_t,quda::plus<typename R::reduce_t>>,void>
blockReduce(sycl::group<3> grp, T &out, const T &in, const R &)
{
  blockReduceSum(grp, out, in);
}

template <typename T, typename R>
inline std::enable_if_t<std::is_same_v<typename R::reducer_t,quda::maximum<typename R::reduce_t>>,void>
blockReduce(sycl::group<3> grp, T &out, const T &in, const R &)
{
  blockReduceMax(grp, out, in);
}

template <typename T, typename R>
inline std::enable_if_t<std::is_same_v<typename R::reducer_t,quda::minimum<typename R::reduce_t>>,void>
blockReduce(sycl::group<3> grp, T &out, const T &in, const R &)
{
  blockReduceMin(grp, out, in);
}
