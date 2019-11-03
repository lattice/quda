#ifndef HELPER_FORWARDER
#define HELPER_FORWARDER

#include <type_traits>
#include <utility>

namespace
{
    // TODO: temporary.
    template<typename F>
    class Forwarder {
        F f_;

        template<typename T>
        __device__
        static
        constexpr
        T&& forward_(typename std::remove_reference<T>::type& x)
        {
            return static_cast<T&&>(x);
        }

        template<typename T>
        __device__
        static
        constexpr
        T&& forward_(typename std::remove_reference<T>::type&& x)
        {
            return static_cast<T&&>(x);
        }
    public:
        explicit
        Forwarder(F f) : f_{std::move(f)} {}

        template<typename... Ts>
        __device__
        void operator()(Ts&&... xs) const { f_(forward_<Ts>(xs)...); }
    };

    template<typename F>
    static
    inline
    Forwarder<F> make_forwarder(F f)
    {
        return Forwarder<F>{std::move(f)};
    }
}
#endif
