#pragma once

namespace detail {

template<typename T, int m>
struct array {
    typedef T value_type;
    typedef T head_type;
    typedef array<T, m-1> tail_type;
    static const int size = m;
    head_type head;
    tail_type tail;
    __host__ __device__ __forceinline__
    array(head_type h, const tail_type& t) : head(h), tail(t) {}
    __host__ __device__ __forceinline__
    array() : head(), tail() {}
    __host__ __device__ __forceinline__
    array(const array& other) : head(other.head), tail(other.tail) {}
    __host__ __device__ __forceinline__
    array& operator=(const array& other) {
        head = other.head;
        tail = other.tail;
        return *this;
    }
    __host__ __device__ __forceinline__
    bool operator==(const array& other) const {
        return (head == other.head) && (tail == other.tail);
    }
    __host__ __device__ __forceinline__
    bool operator!=(const array& other) const {
        return !operator==(other);
    }
};

template<typename T>
struct array<T, 1> {
    typedef T value_type;
    typedef T head_type;
    static const int size = 1;
    head_type head;
    __host__ __device__ __forceinline__
    array(head_type h) : head(h){}
    __host__ __device__ __forceinline__
    array() : head() {}
    __host__ __device__ __forceinline__
    array(const array& other) : head(other.head) {}
    __host__ __device__ __forceinline__
    array& operator=(const array& other) {
        head = other.head;
        return *this;
    }
    __host__ __device__ __forceinline__
    bool operator==(const array& other) const {
        return (head == other.head);
    }
    __host__ __device__ __forceinline__
    bool operator!=(const array& other) const {
        return !operator==(other);
    }
};

template<typename T>
struct array<T, 0>{};

template<typename T, int m, int i>
struct get_impl {
    __host__ __device__ __forceinline__ static T& impl(array<T, m>& src) {
        return get_impl<T, m-1, i-1>::impl(src.tail);
    }
    __host__ __device__ __forceinline__ static T impl(const array<T, m>& src) {
        return get_impl<T, m-1, i-1>::impl(src.tail);
    }
};

template<typename T, int m>
struct get_impl<T, m, 0> {
    __host__ __device__ __forceinline__ static T& impl(array<T, m>& src) {
        return src.head;
    }
    __host__ __device__ __forceinline__ static T impl(const array<T, m>& src) {
        return src.head;
    }
};

template<int i, typename T, int m>
__host__ __device__ __forceinline__
T& get(array<T, m>& src) {
    return detail::get_impl<T, m, i>::impl(src);
}

template<int i, typename T, int m>
__host__ __device__ __forceinline__
T get(const array<T, m>& src) {
    return detail::get_impl<T, m, i>::impl(src);
}
  
} //end namespace detail
