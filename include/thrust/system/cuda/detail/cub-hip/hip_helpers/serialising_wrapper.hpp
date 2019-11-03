#pragma once

#include <hip/hip_runtime.h>

#include <type_traits>
#include <utility>

namespace
{
    // TODO: temporary!
    template<typename T>
    struct Wrapper {
        char d_[sizeof(T)];

        __host__ __device__
        Wrapper(const T& x)
        {
            const char* it = reinterpret_cast<const char*>(&x);
            for (auto i = 0u; i != sizeof(T); ++i) {
                d_[i] = it[i];
            }
        }

        __host__ __device__
        Wrapper(const Wrapper& x)
        {
            for (auto i = 0u; i != sizeof(T); ++i) d_[i] = x.d_[i];
        }

        __host__ __device__
        operator T() const { return reinterpret_cast<const T&>(d_); }

        template<typename... Args>
        __host__ __device__
        auto operator()(Args&&... args) -> decltype(std::declval<T&>()(std::forward<Args>(args)...))
        {
            return reinterpret_cast<T&>(d_)(std::forward<Args>(args)...);
        }

        template<typename... Args>
        __host__ __device__
        auto operator()(Args&&... args) const -> decltype(std::declval<const T&>()(std::forward<Args>(args)...))
        {
            return reinterpret_cast<const T&>(d_)(std::forward<Args>(args)...);
        }

        template<typename U>
        __host__ __device__
        auto operator[](U&& x) -> decltype(std::declval<T&>()[std::forward<U>(x)])
        {
            return reinterpret_cast<T&>(d_)[std::forward<U>(x)];
        }

        template<typename U>
        __host__ __device__
        auto operator[](U&& x) const -> decltype(std::declval<const T&>()[std::forward<U>(x)])
        {
            return reinterpret_cast<const T&>(d_)[std::forward<U>(x)];
        }
    };
}