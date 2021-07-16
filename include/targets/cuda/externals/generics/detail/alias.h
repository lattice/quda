#pragma once
#include <generics/detail/array.h>

namespace detail {

template<typename T, int p>
struct size_multiple_power_of_two {
    static const bool value = (sizeof(T) & ((1 << p) - 1)) == 0;
};

template<typename T,
         bool use_int=size_multiple_power_of_two<T, 2>::value>
struct working_type {
    typedef char type;
};

template<typename T>
struct working_type<T, true> {
    typedef int type;
};


template<typename T, typename U>
struct aliased_size {
    static const int value = sizeof(T)/sizeof(U);
};

template<typename T>
struct working_array {
    typedef typename working_type<T>::type U;
    static const int r = aliased_size<T, U>::value;
    typedef array<U, r> type;
};

template<typename T,
         typename U=typename working_type<T>::type,
         int r=aliased_size<T, U>::value>
struct dismember {
    typedef array<U, r> result_type;
    static const int idx = aliased_size<T, U>::value - r;
    __host__ __device__ __forceinline__
    static result_type impl(const T& t) {
        return result_type(((const U*)&t)[idx],
                           dismember<T, U, r-1>::impl(t));
    }
};

template<typename T, typename U>
struct dismember<T, U, 1> {
    typedef array<U, 1> result_type;
    static const int idx = aliased_size<T, U>::value - 1;
    __host__ __device__ __forceinline__
    static result_type impl(const T& t) {
        return result_type(((const U*)&t)[idx]);
    }
};

template<typename U, typename T>
__host__ __device__ __forceinline__
array<U, detail::aliased_size<T, U>::value> lyse(const T& in) {
    return detail::dismember<T, U>::impl(in);
}


template<typename T,
         typename U=typename working_type<T>::type,
         int r=aliased_size<T, U>::value>
struct remember {
    static const int idx = aliased_size<T, U>::value - r;
    __host__ __device__ __forceinline__
    static void impl(const array<U, r>& d, T& t) {
        ((U*)&t)[idx] = d.head;
        remember<T, U, r-1>::impl(d.tail, t);
    }
};

template<typename T, typename U>
struct remember<T, U, 1> {
    static const int idx = aliased_size<T, U>::value - 1;
    __host__ __device__ __forceinline__
    static void impl(const array<U, 1>& d, const T& t) {
        ((U*)&t)[idx] = d.head;
    }
};


template<typename T>
__host__ __device__ __forceinline__
T fuse(const typename working_array<T>::type& in) {
    T result;
    typedef typename working_type<T>::type U;
    remember<T, U>::impl(in, result);
    return result;
}

}

