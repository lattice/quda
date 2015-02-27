/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2014, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/util_allocator.cuh>
#include <cub/util_device.cuh>

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_reduce.cuh>

#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_scan.cuh>

#include <cub/device/device_histogram.cuh>

#include <cub/iterator/transform_input_iterator.cuh>
#include <cub/iterator/tex_obj_input_iterator.cuh>

#include <cuda_profiler_api.h>

/// Optional outer namespace(s)
CUB_NS_PREFIX

/// CUB namespace
namespace cub {

template <int N>
struct Log2RoundDown
{
    enum {VALUE = Log2<N>::VALUE - (PowerOfTwo<N>::VALUE ? 0 : 1) };
};

template <int N>
struct RoundUpToPowerOfTwo
{
    enum {VALUE = 1 << Log2<N>::VALUE };
};

/**
 * \brief Fixed-size vector that is based on statically-addressed items.
 *
 * When the vector is dynamically indexed, it performs a runtime search for the right item.
 *
 */
template<typename T, int N> struct RegVector;

// Base case in a recursive definition
template<typename T> struct RegVector<T,1> {
    enum {
        LENGTH = 1,
        SIZE_BYTES = sizeof(T)
    };
    typedef T Type;
    typedef T BaseType;
    T value;
    __host__ __device__ __forceinline__
    RegVector() {}
    __host__ __device__ __forceinline__
    RegVector(const T init[1]) : value(init[0]) {}
    __host__ __device__ __forceinline__
    RegVector(const T& fill) : value(fill) {}
    __host__ __device__ __forceinline__
    RegVector(const T& first, const T& default_val) : value(first) {}
    __host__ __device__ __forceinline__
    T operator[](int idx) const { assert(idx == 0); return value; }
    __host__ __device__ __forceinline__
    void set(int i, T val) { if (i == 0) value = val; }
    template<int I> __host__ __device__ __forceinline__
    T& operator[](Int2Type<0> idx) { return value; }
    template<int I> __host__ __device__ __forceinline__
    const T& operator[](Int2Type<0> idx) const { return value; }
};

template<typename T, int N> struct RegVector : public RegVector<T,N-1>
{
    enum {
        LENGTH = N,
        SIZE_BYTES = N * sizeof(T)
    };
    typedef T Type;
    typedef T BaseType;
    T value;
    __host__ __device__ __forceinline__
    RegVector() {}
    __host__ __device__ __forceinline__
    RegVector(const T init[N]) : RegVector<T,N-1>(init), value(init[N-1]) {}
    __host__ __device__ __forceinline__
    RegVector(const T& fill) : RegVector<T,N-1>(fill), value(fill) {}
    __host__ __device__ __forceinline__
    RegVector(const T& first, const T& default_val) : RegVector<T,N-1>(first, default_val), value(default_val) {}

    __host__ __device__ __forceinline__ T operator[] (int idx) const {
        if (idx == N-1)
            return value;
        else
            return (*(RegVector<T,N-1>*)this)[idx];
    }

    __host__ __device__ __forceinline__
    void set(int i, T val)
    {
        if (i == N-1) value = val;
        else ((RegVector<T,N-1>&)(*this)).set(i, val);
    }

    template<int I> __host__ __device__ __forceinline__ T& operator[](Int2Type<I> idx) { return ((RegVector<T,I+1>*)this)->value; }
    template<int I> __host__ __device__ __forceinline__ const T& operator[](Int2Type<I> idx) const { return ((RegVector<T,I+1>*)this)->value; }
};

struct ExtremeFlags
{
    int nan;
//    char inf[2];    //< infinity [+,-]
//    char nan;       //< nan
//    char _pad;
};


template<typename Tdest, typename Tsrc> __host__ __device__ __forceinline__
Tdest reinterpret_bits (const Tsrc& from)
{
    union {
        Tsrc t1;
        Tdest t2;
    } val;
    val.t1 = from;
    return val.t2;
}

/**
 * \brief Accumulates doubles with extended precision into a set of double words.
 *
 * \tparam Expansions           Number of words to store the sum
 *
 *
 * Example:
 *
 * AccumulatorDouble<2> accum(0.0);
 * accum.Add(1e-12);
 * accum.Add(1e+12);
 * accum.Add(1.0);
 * double a = accum[0];
 * double b = accum[1];
 * accum.print();
 *
 */
template<int Expansions>
struct AccumulatorDouble
{
    enum { SIZE = Expansions };

    typedef AccumulatorDouble<Expansions> Type;
    typedef RegVector<double,Expansions> TVec;

    typedef Int2Type<1> DEFAULT_FIX_INFTY;

    TVec _vec;  //vector of SIZE doubles stored as named registers. _vec[0] is the most significant word.

    __host__ __device__ __forceinline__ AccumulatorDouble() {}                                   //< uninitialized
    __host__ __device__ __forceinline__ AccumulatorDouble(double fill) : _vec(fill) {}          //< fill all words with a value
    __host__ __device__ __forceinline__ AccumulatorDouble(double first, double default_val)    //< sets word 0 to first, and all the others to default_val
        : _vec(first, default_val) {}

    /**
     * \brief Loads values from an array
     */
    __host__ __device__ __forceinline__ void Load(const double vals[SIZE])
    {
        _vec = TVec(vals);
    }

    /**
     * \brief Stores values to an array
     */
    __host__ __device__ __forceinline__ void Store(double vals[SIZE]) const
    {
#pragma unroll
        for (int i = 0; i < SIZE; i++)
        {
            vals[i] = _vec[i];
        }
    }

    /**
     * \brief Adds a double-precision floating point value to the result.
     *
     * Returns the remainder that could not be saved in the accumulator
     */
    __host__ __device__ __forceinline__ double Add(const double &v)
    {
        return add(v, DEFAULT_FIX_INFTY());
    }

    template <int FIX_INFTY>
    __host__ __device__ __forceinline__ double Add(const double &v)
    {
        return add(v, Int2Type<FIX_INFTY>());
    }

    /**
     * \brief Adds the value of another accumulator to this one
     */
    __host__ __device__ __forceinline__ void Add(const Type &v)
    {
        add(v, DEFAULT_FIX_INFTY());
    }

    template <int FIX_INFTY>
    __host__ __device__ __forceinline__ void Add(const Type &v)
    {
        add(v, Int2Type<FIX_INFTY>());
    }

    /**
     * \brief Serially adds the values of an array to the accumulator
     */
    __host__ __device__ __forceinline__ void Add(const double* arr, int len)
    {
        Add< DEFAULT_FIX_INFTY::VALUE >(arr,len);
    }

    template<int FIX_INFTY>
    __host__ __device__ __forceinline__ void Add(const double* arr, int len)
    {
        if (len < 0)
        {
            return;
        }

        for (int i = 0; i < len; i++)
        {
            add(arr[i], Int2Type<FIX_INFTY>());
        }
    }

    /**
     * \brief Serially adds the values of a fixed-size array to the accumulator
     */
    template<int LENGTH>
    __host__ __device__ __forceinline__ void Add(const double* arr)
    {
        Add< LENGTH, DEFAULT_FIX_INFTY::VALUE >(arr);
    }

    template<int LENGTH, int FIX_INFTY>
    __host__ __device__ __forceinline__ void Add(const double* arr)
    {
#pragma unroll
        for (int i = 0; i < LENGTH; i++)
        {
            add(arr[i], Int2Type<FIX_INFTY>());
        }
    }

    /**
     * \brief Adds two accumulators. Makes this class a functor for adding accumulators.
     */
    __host__ __device__ __forceinline__ Type operator()(const Type &a, const Type &b) const
    {
        Type sum = a;
        sum.Add(b);
        return sum;
    }

    /**
     * \brief Returns the value at a given index.
     *
     * Supports runtime and compile-time indexing.
     */
    __host__ __device__ __forceinline__ double operator[](int i) const { return _vec[i]; }
    __host__ __device__ __forceinline__ void set(int i, const double& val) { _vec.set(i,val); }
    template<int INDEX> __host__ __device__ __forceinline__ double& operator[](Int2Type<INDEX> i) { return _vec[i]; }
    template<int INDEX> __host__ __device__ __forceinline__ const double& operator[](Int2Type<INDEX> i) const { return _vec[i]; }

    /**
     * \brief Removes overlap between words by running them through a new accumulator
     */
    __host__ __device__ __forceinline__ void Normalize()
    {
        Type tmp(_vec[0], 0.);
#pragma unroll
        for (int i = 1; i < SIZE; i++)
        {
            tmp.Add(_vec[i]);
        }
        _vec = tmp._vec;
    }

    /**
     * \brief Prints the values of the accumulation vector to stdout
     */
    __host__ __device__ __forceinline__ void print() const
    {
        double words[SIZE];
        Store(words);
        printf("[ ");
        if (SIZE > 0)
        {
            printf("%g", words[0]);
#if CUB_PTX_ARCH == 0
            printf(" [0x%016llX] ", reinterpret_bits<unsigned long long>(words[0]));
#else
            printf(" [0x%016llX] ", __double_as_longlong(words[0]));
#endif
        }
        for(int i=1; i<SIZE; i++)
        {
            printf(", %g", words[i]);
            printf(" [0x%016llX] ", reinterpret_bits<unsigned long long>(words[i]));
        }
        printf(" ]");
    }

    /**
     * \brief Applies a functor to each element of the accumulation vector
     *
     * \par Snippet
     * The code snippet below illustrates a possible use of the function to print
     * the values.
     * \par
     * \code
     * struct printy {
     *     __host__ __device__ void operator()(double& d)
     *     {
     *         printf("{%g} ", d);
     *     }
     * };
     *... later in the code ...
     * printy op;
     * normalizer.ForEachWord(op);
     *
     * \endcode
     *
     */
    template<typename OP>
    __host__ __device__ __forceinline__ void ForEachWord(OP& op)
    {
        ForEachWord(op, Int2Type<0>());
    }

protected:
    __host__ __device__ __forceinline__
    double twoSum(double a, double b, double& rem)
    {
    #if 0
        // works only for positive numbers
        double mn = min(a,b);
        double mx = max(a,b);
        return quickTwoSum(mx,mn,rem);

        // always works but performance is worse
        //return (fabs(a) >= fabs(b) ? quickTwoSum(a,b,rem) : quickTwoSum(b,a,rem));
    #else
        double s, v;
        s = a + b;
        v = s - a;
        rem = (a - (s - v)) + (b - v);
        return s;
    #endif
    }

    __host__ __device__ __forceinline__
    double quickTwoSum(double a, double b, double& rem)
    {
        double s;
        s = a + b;
        rem = b - (s - a);
        return s;
    }

    __host__ __device__ __forceinline__
    RegVector<double, 2> twoSum(double a, double b)
    {
        double s, r;
        s = twoSum(a, b, r);
        return RegVector<double, 2>(s, r);
    }


    __host__ __device__ __forceinline__
    RegVector<double, 2> quickTwoSum(double a, double b)
    {
        double s, r;
        s = quickTwoSum(a, b, r);
        return RegVector<double, 2>(s, r);
    }

    template<int FIX_INFTY>
    __host__ __device__ __forceinline__ void add(const Type& v, Int2Type<FIX_INFTY> fix_infty)
    {
#pragma unroll
        for (int i = 0; i < SIZE; i++)
        {
            double rem = add(v[i], fix_infty);
            if (rem != 0.0)
            {
                Normalize();
                add(rem, fix_infty);
            }
        }
    }

    template<int FIX_INFTY>
    __host__ __device__ __forceinline__ double add(const double& v, Int2Type<FIX_INFTY> fix_infty)
    {
        // TODO: tune with/without break for best performance
        const bool short_circuit = true;
        double rem = v;                     // remainder
#pragma unroll
        for(int i = 0; i < SIZE; i++)
        {
            if (!short_circuit || rem != 0.0)
            {
                _vec.set(i, twoSum(_vec[i], rem, rem));

                if (FIX_INFTY)
                {
                    if (isinf(_vec[i]))
                    {
                        rem = 0.0;
                    }
                }
            }
        }
        return rem;
    }

    __host__ __device__ __forceinline__ TVec fix_inf(TVec accum)
    {
        if (isinf(accum[0]))
        {
            return TVec(accum[0], 0.0);
        }
        return accum;
    }
};


/**
 * Specialized more efficient version for 2-wide accumulators
 */

template<>
__host__ __device__ __forceinline__ RegVector<double,2> AccumulatorDouble<2>::fix_inf(RegVector<double,2> accum)
{
    RegVector<double,2> out = accum;
    out.set(1,out[1] * (0 == isinf(out[0])));
    return out;
}

template<>
template<int FIX_INFTY>
__host__ __device__ __forceinline__ double AccumulatorDouble<2>::add<FIX_INFTY>(const double& b, Int2Type<FIX_INFTY>)
{
    Int2Type<1> I1;
    const RegVector<double, 2>& a = _vec;
    RegVector<double, 2> s, t;
    s = twoSum(a[0], b);
    s[I1] += a[1];
    if (FIX_INFTY)
        fix_inf(s);
    s = quickTwoSum(s[0], s[1]);
    _vec = s;
    return 0.;
}

template<>
template<int FIX_INFTY>
__host__ __device__ __forceinline__ void AccumulatorDouble<2>::add<FIX_INFTY>(const AccumulatorDouble<2>& other, Int2Type<FIX_INFTY>)
{
    // OPTIMIZE: instead of running fix_inf, raise an +inf/-inf flag if isinf is true. If flag is up, ignore reduction result.
    //    if (other[1] == 0.0)
    //    {
    //        add(other[0]);
    //    }
    Int2Type<1> I1;
    const RegVector<double, 2>& a = _vec;
    RegVector<double, 2> b = other._vec;
    RegVector<double, 2> s, t;
    s = twoSum(a[0], b[0]);
    if (FIX_INFTY)
        s = fix_inf(s);
    t = twoSum(a[1], b[1]);
    s[I1] += t[0];
    s = quickTwoSum(s[0], s[1]);
    if (FIX_INFTY)
        s = fix_inf(s);
    s[I1] += t[1];
    s = quickTwoSum(s[0], s[1]);
    if (FIX_INFTY)
        s = fix_inf(s);
    _vec = s;
}

/*
 * sum operator for accumulators
 */
template<int Expansion>
__host__ __device__ __forceinline__
AccumulatorDouble<Expansion> operator+(
    const AccumulatorDouble<Expansion>& a,
    const AccumulatorDouble<Expansion>& b)
{
    AccumulatorDouble<Expansion> sum = a;
    sum.Add(b);
    return sum;
}

/**
 * Computes the binning configuration based on setup arguments at compile time.
 */
template <
    int         BLOCK_DIM_X,
    int         ITEMS_PER_THREAD,
    int         EXPANSIONS,
    int         RADIX_SORT_BITS             = 4,
    int         BLOCK_DIM_Y                  = 1,
    int         BLOCK_DIM_Z                  = 1>
struct AccumulatorBinsMetadata
{
private:
    /* auxiliary computations */
    struct _temp {
        enum {
            _MAX_POW2_ENUM = 1 << (sizeof(int) * 8 - 2),                                                     //< maximum power-of-two enum. enum values must be representable as an int
            _DOUBLE_MANTISSA_BITS   = 52,
            _DOUBLE_EXPONENT_BITS   = 11,
            _BLOCK_THREADS          = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
            _ITEMS_PER_ITERATION    = _BLOCK_THREADS * ITEMS_PER_THREAD,                                    //< number of items reduced in each iteration
            _MAX_OVERFLOW_FREQUENCY = 256,                                                                    //< minimum number of iterations before an overflow could happen
            _EXTENDED_MANTISSA_BITS = _DOUBLE_MANTISSA_BITS * (EXPANSIONS - 1),                              //< extra mantissa bits due to using multiple double words
            _MAX_EXPONENT_BITS_PER_ACCUM = Log2RoundDown<_EXTENDED_MANTISSA_BITS>::VALUE,                     //< maximum number of exponent bits that can be covered by a single accumulator
            _LOG_ITEMS_PER_ACCUM_LOW  = _EXTENDED_MANTISSA_BITS - (1 << _MAX_EXPONENT_BITS_PER_ACCUM),        //< (log2) for max number of exponent bits, the number of items that can be added to an accumulator before it may overflow
            _LOG_ITEMS_PER_ACCUM_MIN  = Log2<_MAX_OVERFLOW_FREQUENCY * _ITEMS_PER_ITERATION>::VALUE,          //< (log2) minimum allowed number of items that can be added to an accumulator before it may overflow
            _EXPONENT_BITS_PER_BIN   = _MAX_EXPONENT_BITS_PER_ACCUM - (_LOG_ITEMS_PER_ACCUM_LOW >= _LOG_ITEMS_PER_ACCUM_MIN ? 0 : 1),  //< each bin covers a range of numbers that have the same lower X exponent bits
            _NUM_BINS                = 1 << (_DOUBLE_EXPONENT_BITS - _EXPONENT_BITS_PER_BIN),                 //< number of bins
            _LOG_BIN_CAPACITY        = _EXTENDED_MANTISSA_BITS - (1 << _EXPONENT_BITS_PER_BIN),               //< (log2) maximum number of items that can be added to a bin before it could overflow
            _BIN_CAPACITY_TRUNC      = 1 << CUB_MIN(_LOG_BIN_CAPACITY, Log2<_MAX_POW2_ENUM>::VALUE),          //< number of items that can be added to a bin, truncated by the max power-of-two enum
            _BIN_SIZE_BYTES          = sizeof(AccumulatorDouble<EXPANSIONS>),                               //< (log2) maximum integer. used to prevent enum overflow
            _SORT_BITS               = _DOUBLE_EXPONENT_BITS - _EXPONENT_BITS_PER_BIN,                        //< number of exponent bits to sort
            _RADIX_BITS              = CUB_MIN(RADIX_SORT_BITS, _SORT_BITS),                                  //< radix bits for sorting
        };
    };
public:
    typedef AccumulatorDouble<EXPANSIONS> BinType;

    enum {
        EXPONENT_BITS_PER_BIN   = _temp::_EXPONENT_BITS_PER_BIN,  //< items with the same lower X exponent bits are accumulated to the same bin
        NUM_BINS                = _temp::_NUM_BINS,               //< number of bins
        BIN_CAPACITY            = _temp::_BIN_CAPACITY_TRUNC,     //< maximum number of items that can be accumulated in a bin before it could overflow
        BIN_SIZE_BYTES          = _temp::_BIN_SIZE_BYTES,         //< size of accumulator (bin) in bytes
        RADIX_BITS              = _temp::_RADIX_BITS,             //< radix bits for sorting

    };
    __host__ __device__
    static void info()
    {
        printf(
            "EXPONENT_BITS_PER_BIN  = %d \n"
            "NUM_BINS               = %d \n"
            "BIN_CAPACITY           = %d \n"
            "BIN_SIZE_BYTES         = %d \n",
            EXPONENT_BITS_PER_BIN,
            NUM_BINS             ,
            BIN_CAPACITY         ,
            BIN_SIZE_BYTES
            );
    }
};

/**
 * DeviceAccurateSum provides operations to accurately sum an array of doubles without round-off error
 */
template <
    int         BLOCK_DIM_X                  = 64,
    int         ITEMS_PER_THREAD            = 2,
    int         EXPANSIONS                   = 2,
    int         RADIX_BITS                   = 4,
    int         BLOCK_DIM_Y                  = 1,
    int         BLOCK_DIM_Z                  = 1
    >
class DeviceAccurateSum
{
public:
    enum {
        BLOCK_THREADS           = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,
        DOUBLE_MANTISSA_BITS    = 52,
        DOUBLE_EXPONENT_BITS    = 11,
        ITEMS_PER_THREAD_TAIL = 1,    //< for remaining items after completing all the full tiles
    };

    typedef AccumulatorBinsMetadata<BLOCK_DIM_X, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, BLOCK_DIM_Y, BLOCK_DIM_Z> Meta;
    typedef NumericTraits<double>::UnsignedBits UnsignedBits;
    typedef int   BinIdT;
    typedef int   FlagT;      // also used for counter
    typedef AccumulatorDouble<EXPANSIONS> Accumulator;

    // A structure that binds bin-id and accumulator, to be used in scan
    struct AccumBinPair
    {
        Accumulator             accum;
        BinIdT                  bin;
        int                     _pad;       //< pad struct to multiple of 8bytes
    };

    static const bool CONFIG_SORT_MEMOIZE = ((CUB_PTX_ARCH >= 350) ? true : false);
    typedef BlockLoad<double*, BLOCK_THREADS, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE> BlockLoad;
    typedef BlockRadixSort<
        UnsignedBits, BLOCK_THREADS, ITEMS_PER_THREAD, NullType, Meta::RADIX_BITS,
        CONFIG_SORT_MEMOIZE, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeEightByte> BlockRadixSortFullTiles;
    typedef BlockRadixSort<
            UnsignedBits, BLOCK_THREADS, ITEMS_PER_THREAD_TAIL, NullType, Meta::RADIX_BITS,
            CONFIG_SORT_MEMOIZE, BLOCK_SCAN_WARP_SCANS, cudaSharedMemBankSizeEightByte> BlockRadixSortTail;
    typedef BlockDiscontinuity<BinIdT, BLOCK_THREADS> BlockDiscontinuity;
    typedef BlockScan<AccumBinPair, BLOCK_THREADS, BLOCK_SCAN_WARP_SCANS> BlockScan;

    /// Shared memory storage layout type for DeviceAccurateSum::SumToBins()
    // The temp space for sort is typically larger than for load, flag, and scan combined
    struct _TempStorage
    {
        union
        {
            union
            {
                typename BlockRadixSortFullTiles::TempStorage  sort;
                typename BlockRadixSortTail::TempStorage       sort_tail;
            };
            struct {
//                typename BlockLoad::TempStorage          load;
                typename BlockDiscontinuity::TempStorage flag;
                typename BlockScan::TempStorage          scan;
            };
        };
        ExtremeFlags extreme_flags;
    };

    _TempStorage &temp_storage;

    __device__ __forceinline__ _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }

//public:
    struct TempStorage : Uninitialized<_TempStorage> {};

    __device__ __forceinline__ DeviceAccurateSum()
    :
        temp_storage(PrivateStorage())
    {}

    __device__ __forceinline__ DeviceAccurateSum(
        TempStorage &temp_storage)
    :
        temp_storage(temp_storage.Alias())
        {}

    /**
     * Defines
     */
    template<typename    InputIteratorT>
    __device__ __forceinline__ void SumToBins(
        InputIteratorT  d_in,                 //< [in]  input array
        int             num_items,            //< [in]  input array size
        void           *d_accumulators,       //< [out] accumulator bins
        size_t          accumulators_bytes,   //< [in]  size of accumulator bins array in bytes
        void           *d_global_bin_set,
        size_t          global_bins_bytes,
        ExtremeFlags    *d_extreme_flags)
    {
        if (num_items == 0)
        {
            return;             //< for warmup
        }

        if (d_extreme_flags->nan)   //< if nan flag already marked then abort. result is nan.
        {
            return;
        }

        ///////////////////////////////////////////////////////////////////////
        __shared__ Accumulator bins[Meta::NUM_BINS];
        __shared__ int s_FIX_INFTY;
        ///////////////////////////////////////////////////////////////////////

        enum {
            TILE_SIZE              = (BLOCK_THREADS * ITEMS_PER_THREAD),
        };

        /// INITIALIZE BINS IN SHARED MEMORY
        asm("// INIT SMEM {");
        const int COUNT = Meta::NUM_BINS * Meta::BIN_SIZE_BYTES / sizeof(double);
        InitSmemAsync<COUNT, double>((double*)bins, 0.0);
        s_FIX_INFTY = 0;
        asm("// INIT SMEM }");

        if (threadIdx.x == 0)
        {
            temp_storage.extreme_flags.nan = 0;
        }

        // add offset that depends on block index
        int tiles_in_grid = gridDim.x * TILE_SIZE;
        int num_items_full_tiles = CUB_ROUND_DOWN_NEAREST(num_items, tiles_in_grid);
        int num_remaining_items = num_items - num_items_full_tiles;

        Int2Type<1> full_tiles;
        Int2Type<0> non_full_tiles;

        SumToBins_internal(d_in+0,                    d_in+num_items_full_tiles, num_items_full_tiles, d_extreme_flags, bins, s_FIX_INFTY, full_tiles);
        SumToBins_internal(d_in+num_items_full_tiles, d_in+num_items,            num_remaining_items,  d_extreme_flags, bins, s_FIX_INFTY, non_full_tiles);


        /// STORE BINS TO GLOBAL MEM
        StoreBinsToGlobalMem(d_accumulators, bins);

        s_FIX_INFTY = 0;

        if (threadIdx.x == 0)
        {
            if (temp_storage.extreme_flags.nan)
            {
                d_extreme_flags->nan = 1;
            }

        }
    }

    template<typename    InputIteratorT, int FULL_TILES>
    __device__ __forceinline__ void SumToBins_internal(
        InputIteratorT  d_in_begin,
        InputIteratorT  d_in_end,
        int             num_items,
        ExtremeFlags   *d_extreme_flags,
        Accumulator   (&bins)[Meta::NUM_BINS],    /*__shared__*/
        int            &s_FIX_INFTY,              /*__shared__*/
        Int2Type<FULL_TILES> oob_check
        )
    {
        enum {
            /* The default ITEMS_PER_THREAD is overridden with 1 for non-full tiles */
            _ITEMS_PER_THREAD      = (FULL_TILES ? ITEMS_PER_THREAD : ITEMS_PER_THREAD_TAIL),
            TILE_SIZE              = (BLOCK_THREADS * _ITEMS_PER_THREAD),
            EXP_BIT_SORT_BEGIN     = (DOUBLE_MANTISSA_BITS + Meta::EXPONENT_BITS_PER_BIN),
            EXP_BIT_SORT_END       = (DOUBLE_MANTISSA_BITS + DOUBLE_EXPONENT_BITS),
        };

        typedef typename If<FULL_TILES, BlockRadixSortFullTiles, BlockRadixSortTail>::Type BlockRadixSort;

        double items[_ITEMS_PER_THREAD];

        int tiles_per_block = num_items / (gridDim.x * TILE_SIZE) + (FULL_TILES ? 0 : 1);
        InputIteratorT d_in = d_in_begin + blockIdx.x * TILE_SIZE;

        /// PROCESS INPUT

        // Loop over tiles
        #pragma unroll 4
        for (int tile = 0; tile < tiles_per_block; tile++)
        {
            // BINS ARE BEING UPDATED
            // ITEMS ARE NOT BEING USED
            // Load tile
//            BlockLoad(temp_storage.load).Load(d_in, items);
//            __syncthreads();
            asm("// Load items {");
            // Load items in direct striped order
            #pragma unroll
            for (int ITEM = 0; ITEM < _ITEMS_PER_THREAD; ITEM++)
            {
                if (FULL_TILES ||
                    IsValidOutOfBoundsAccess<InputIteratorT>::VALUE )   //< out-of-bounds array accesses are assumed to return 0.0
                    items[ITEM] = d_in[ITEM * BLOCK_THREADS + threadIdx.x];
                else
                {
                    items[ITEM] = 0.0;
                    InputIteratorT in_ptr = d_in + (ITEM * BLOCK_THREADS + threadIdx.x);
                    if (d_in_end - in_ptr > 0)
                        items[ITEM] = *in_ptr;
                }
            }

            asm("// Load items }");
            // BINS ARE NOT BEING UPDATED
            // ITEMS ARE BEING USED

            // Reduce values and update bins
            ///////////////////////////////////////////////

            /// RADIX SORT BY (SOME) EXPONENT BITS
            asm("// SORT {");
            __syncthreads();
            // SHARED MEM IN USE FOR SORT
            UnsignedBits (*cvt_to_ubits)[_ITEMS_PER_THREAD] = (UnsignedBits(*)[_ITEMS_PER_THREAD])items;
            BlockRadixSort((typename BlockRadixSort::TempStorage&)(temp_storage.sort))
                .Sort(*cvt_to_ubits,EXP_BIT_SORT_BEGIN,EXP_BIT_SORT_END);
            asm("// SORT }");

            /// REDUCE-BY-KEY (SETUP)
            asm("// REDUCE-BY-KEY {");
            BinIdT          bin_ids     [_ITEMS_PER_THREAD];
            FlagT           tail_flags  [_ITEMS_PER_THREAD];
            AccumBinPair    zip         [_ITEMS_PER_THREAD];
            AccumBinPair    zipout      [_ITEMS_PER_THREAD];
            #pragma unroll
            for (int i = 0; i < _ITEMS_PER_THREAD; ++i)
            {
                if (isnan(items[i]))
                {
                    temp_storage.extreme_flags.nan = 1;
                }

                const double BIG_DBL = 1e306;
                if (abs(items[i]) > BIG_DBL)
                {
                    s_FIX_INFTY = 1;
                }

                bin_ids[i] = bin_id(items[i]);
                zip[i].accum = Accumulator(items[i], 0.0);
                zip[i].bin = bin_ids[i];
            }
            __syncthreads();
            // SHARED MEM IN USE FOR SCAN AND FLAGS
            asm("// REDUCE-BY-KEY }");

            // BINS ARE BEING UPDATED
            // ITEMS ARE NOT BEING USED

            /// REDUCE BY KEY
            if (s_FIX_INFTY)
            {
                BlockScan(temp_storage.scan).InclusiveScan(zip, zipout, ReductionOp<1>());
            }
            else
                BlockScan(temp_storage.scan).InclusiveScan(zip, zipout, ReductionOp<0>());
            BlockDiscontinuity(temp_storage.flag).FlagTails(tail_flags, bin_ids, cub::Inequality());

            /// UPDATE BINS
            #pragma unroll
            for (int i = 0; i < _ITEMS_PER_THREAD; ++i)
            {
                if (tail_flags[i])
                {
                    // this is the reduction result for this bin
                    if (1 /*s_FIX_INFTY */ )
                        bins[bin_ids[i]].Add<1>(zipout[i].accum);
                    else
                        bins[bin_ids[i]].Add<0>(zipout[i].accum);
                }
            }

            d_in += gridDim.x * TILE_SIZE;
        }
        __syncthreads();
        // SHARED MEM NOT IN USE
        // BINS ARE NOT BEING UPDATED
    }

//private:

    template<int FIX_INFTY>
    struct ReductionOp
    {
        __device__ __forceinline__ AccumBinPair operator()(
            const AccumBinPair   &first,
            const AccumBinPair   &second)
        {
            AccumBinPair retval = second;
            if (first.bin == second.bin)
            {
                retval.accum.Add<FIX_INFTY>(first.accum);
            }
            return retval;
        }
    };

    template<int COUNT, typename T>
    static __device__ __forceinline__ void InitSmemAsync(T* array, T val)
    {
        T* iptr = array;
        #pragma unroll
        for (int i = 0; i < COUNT / BLOCK_THREADS; i++)
        {
            iptr[i * BLOCK_THREADS + threadIdx.x] = val;
        }
        if (COUNT % BLOCK_THREADS > 0)
        {
            int i = COUNT / BLOCK_THREADS;
            if (i * BLOCK_THREADS + threadIdx.x < COUNT)
            {
                iptr[i * BLOCK_THREADS + threadIdx.x] = val;
            }
        }
    }

    static __device__ __forceinline__ void StoreBinsToGlobalMem(
        void           *d_accumulators,
        Accumulator   (&bins)[Meta::NUM_BINS])    /*__shared__*/
    {
        d_accumulators = (void*)((char*)d_accumulators + blockIdx.x * Meta::NUM_BINS * Meta::BIN_SIZE_BYTES);
        typedef double StoreUnit;
        StoreUnit* isptr = (StoreUnit*)bins;
        StoreUnit* igptr = (StoreUnit*)d_accumulators;
        const int COUNT = Meta::NUM_BINS * Meta::BIN_SIZE_BYTES / sizeof(StoreUnit);
        #pragma unroll
        for (int i = 0; i < COUNT / BLOCK_THREADS; i++)
        {
            double val = isptr[i * BLOCK_THREADS + threadIdx.x];
            igptr[i * BLOCK_THREADS + threadIdx.x] = (isnan(val) ? 0.0 : val);  // TODO: test alternative fix for NaN: fmax(val,0.0) + fmin(val,0.0)
            //                if (!isnan(val))
            //                {
            //                    int bin_id = (i * BLOCK_THREADS + threadIdx.x) / EXPANSIONS;
            //                    atomicAddToBin_< EXPANSIONS + 1 >(((AccumulatorDouble<EXPANSIONS+1>*)d_global_bin_set)[bin_id], val);
            //                }
        }
        if (COUNT % BLOCK_THREADS > 0)
        {
            int i = COUNT / BLOCK_THREADS;
            if (i * BLOCK_THREADS + threadIdx.x < COUNT)
            {
                double val = isptr[i * BLOCK_THREADS + threadIdx.x];
                igptr[i * BLOCK_THREADS + threadIdx.x] = (isnan(val) ? 0.0 : val);
            }
        }
    }

    template<typename Iterator> struct IsValidOutOfBoundsAccess
    {
        enum {VALUE = 0};
    };

    template <typename T, typename OffsetT>
    struct IsValidOutOfBoundsAccess<TexObjInputIterator<T,OffsetT> >
    {
        enum {VALUE = 1};
    };

    static __device__ __forceinline__ BinIdT bin_id(const double& v)
    {
        enum { EXP_BIT_SORT_BEGIN  = (DOUBLE_MANTISSA_BITS + Meta::EXPONENT_BITS_PER_BIN) };

        // maybe with __double2hiint (double x) ...
        // maybe with frexp ...

        unsigned long long llv;
        llv = __double_as_longlong(abs(v));
        int tmp1 = (int)(llv >> EXP_BIT_SORT_BEGIN);
        return (BinIdT)tmp1;
    }

    __device__ double atomicAdd_(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    template <int BIN_EXPANSIONS>
    __device__ void atomicAddToBin_(AccumulatorDouble<BIN_EXPANSIONS>& bin, double val)
    {
        double a, b, s, r, av, bv, ar, br;

        b = val;
        #pragma unroll
        for (int i = 0; i < EXPANSIONS - 1; i++)
        {
            if (b == 0.0) break;
            a = atomicAdd_(&bin[i], b);    // returns a and stores (a+b) in bin[0]
            s = a + b;                              // recompute s=(a+b)
            bv = s - a;
            av = s - bv;
            br = b - bv;
            ar = a - av;
            r = ar + br;
            b = (isnan(r) ? 0.0 : r);
        }
        if (b != 0.0)
        {
            atomicAdd_(&bin[EXPANSIONS - 1], b);      //< don't compute carry for last word
        }
    }
};

template<
    int BLOCK_THREADS,
    int NUM_BINS,
    int NUM_BIN_COPIES
>
struct DeviceAccurateSumSmemAtomic
{
    __device__ void SumToBins(
        double* d_in,
        int num_items,
        double2* d_bins
        )
    {
        // Allocate double2 bins with one double padding to spread bins with the
        // same id across different banks in different binset copies
        __shared__ double s_bins[NUM_BIN_COPIES * (2 * NUM_BINS + 1)];

        double vals[1];
        int nblocks = (int)gridDim.x;
        int iblock = (int)blockIdx.x;
        int ithread = (int)threadIdx.x;
        int items_per_block = num_items / nblocks;
        int bin_group = ithread % NUM_BIN_COPIES;

        double2* mybins = (double2*)(&s_bins[bin_group * (2 * NUM_BINS + 1)]);
        d_in += items_per_block * iblock;

        // initialize bins in shared memory
        const int DBL_BINS_LEN = sizeof(s_bins) / sizeof(double);
        #pragma unroll
        for (int i = threadIdx.x; i < DBL_BINS_LEN; i += BLOCK_THREADS)
        {
            s_bins[i] = 0.0;
        }
        __syncthreads();

        // accumulate in bins
        #pragma unroll 64
        for (int i = threadIdx.x; i < items_per_block; i += BLOCK_THREADS)
        {
            vals[0] = d_in[i];
            int ibin = binid(vals[0]);
            atomicAddToBin(vals[0], ibin, mybins);
        }
        __syncthreads();

        // store bins in global mem
        double* out_bins = ((double*)d_bins) + (iblock * NUM_BIN_COPIES * 2 * NUM_BINS);
        #pragma unroll
        for (int i = threadIdx.x; i < NUM_BINS * 2 * NUM_BIN_COPIES; i += BLOCK_THREADS)
        {
            // place bins with the same id consecutively in output
//            int i_bin = (i / NUM_BIN_COPIES) / 2;
//            int i_word = (i / NUM_BIN_COPIES) % 2;
//            int i_bin_copy = i % NUM_BIN_COPIES;
//            out_bins[i] = bins[i_word + 2 * i_bin + (2 * NUM_BINS + 1) * i_bin_copy];
            int j = i + (i / (2 * NUM_BINS + 1));       //< skip the padding
            out_bins[i] = s_bins[j];
        }
    }       // SumToBins


    // Execute with one thread block
    __device__ void AddTail(
        double* d_tail,
        int tail_len,
        double2* d_bins
        )
    {
        // Allocate double2 bins with one double padding to spread bins with the
        // same id across different banks in different binset copies
        __shared__ double2 s_bins[NUM_BINS];

        double vals[1];

        assert(gridDim.x == 1);

        // load first bin set to smem
        assert(BLOCK_THREADS >= NUM_BINS);
        if (threadIdx.x < NUM_BINS)
        {
            // The bin sets are stored in d_bins with a stripe of NUM_BIN_COPIES bins
            s_bins[threadIdx.x] = d_bins[threadIdx.x];
        }
        __syncthreads();

        // accumulate in bins
        for (int i = threadIdx.x; i < tail_len; i += BLOCK_THREADS)
        {
            vals[0] = d_tail[i];
            int ibin = binid(vals[0]);
            atomicAddToBin(vals[0], ibin, s_bins);
        }
        __syncthreads();

        // store bins back in global mem
        if (threadIdx.x < NUM_BINS)
        {
            d_bins[threadIdx.x] = s_bins[threadIdx.x];
        }
    }       // AddTail

    __device__ int binid(double d)
    {
        enum {
            LOG_NUM_BINS = Log2<NUM_BINS>::VALUE,
            SHIFT_RIGHT = 63 - LOG_NUM_BINS
        };
        long long ll = __double_as_longlong(d);
        int bin = (int)(ll >> SHIFT_RIGHT) & (NUM_BINS-1);
        return bin;
    }

    __device__ double atomicAdd(double* address, double val)
    {
        unsigned long long int* address_as_ull = (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    __device__ void atomicAddToBin(double val, int bin, double2* bins)
    {
        double* dbl_bins = (double*)bins;
        double a, b, x, y, av, bv, ar, br;
        b = val;
        a = atomicAdd(&dbl_bins[2 * bin + 0], b);       // add val (=b) to bin[0] and returns old bin[0] (=a)
        x = a + b;                                      // recompute (a+b)
        bv = x - a;
        av = x - bv;
        br = b - bv;
        ar = a - av;
        y = ar + br;

        if (y != 0.0)
        {
            atomicAdd(&dbl_bins[2 * bin + 1], y);       // add remainder to bin[1]
        }
    }
};

/**
 * \brief Computes an accurate summation of doubles into sets of bins using the sort-reduce method
 *
 * Each thread-block produces one set of bins.
 * Each bin contains the sum of the items in the thread-block's
 *   share of the input that have an exponent in the bin's exponent range.
 *   The bins in each set cover the entire double-precision exponent range.
 */
template <
typename    InputIteratorT,
int         BLOCK_THREADS,
int         ITEMS_PER_THREAD,
int         EXPANSIONS,
int         RADIX_BITS,
int         MIN_CONCURRENT_BLOCKS>
__launch_bounds__ (BLOCK_THREADS, MIN_CONCURRENT_BLOCKS)
__global__ void DeviceAccurateSumKernel(
//    double         *d_in,
    InputIteratorT  d_in,
    int             num_items,
    void           *d_accumulators,
    size_t          accumulators_bytes,
    void           *d_global_bin_set,
    size_t          global_bins_bytes,
    ExtremeFlags    *d_extreme_flags)
{
    typedef DeviceAccurateSum<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS> DeviceAccurateSum;
    __shared__ typename DeviceAccurateSum::TempStorage temp_storage;
    DeviceAccurateSum(temp_storage).SumToBins(d_in, num_items, d_accumulators, accumulators_bytes, d_global_bin_set, global_bins_bytes, d_extreme_flags);
}

/**
 * \brief Computes an accurate summation of doubles into sets of bins using the smem-atomic method
 *
 * Each thread-block produces a number of bin sets.
 * Each bin contains the sum of the items in the thread-block's
 *   share of the input that have an exponent in the bin's exponent range.
 *   The bins in each set cover the entire double-precision exponent range.
 * Assumes that the number of items (num_items) is a multiple of the total number of threads. The
 *   remaining values are later added to the bins by atomicadd_tail_kernel
 */
template<
    int BLOCK_THREADS,
    int BLOCK_WAVE_SIZE,
    int NUM_BINS,
    int NUM_BIN_COPIES
>
__launch_bounds__(BLOCK_THREADS, BLOCK_WAVE_SIZE)
__global__ void atomicadd_kernel(
    double* d_in,
    int num_items,
    double2* d_bins
    )
{
    typedef DeviceAccurateSumSmemAtomic<BLOCK_THREADS, NUM_BINS, NUM_BIN_COPIES> DeviceAccurateSumSmemAtomic;
    DeviceAccurateSumSmemAtomic().SumToBins(d_in, num_items, d_bins);
}

/**
 * \brief Adds tail values that were not processed by atomicadd_kernel.
 *
 * Executed with one thread block with at least NUM_BINS threads.
 * The template parameters
 * Adds the tail values to the first bin set computed by atomicadd_kernel and updates it.
 */
template<
    int BLOCK_THREADS,
    int NUM_BINS
>
__launch_bounds__(BLOCK_THREADS, 1)
__global__ void atomicadd_tail_kernel(
    double* d_tail,
    int tail_len,
    double2* d_bins
    )
{
    typedef DeviceAccurateSumSmemAtomic<BLOCK_THREADS, NUM_BINS, 1> DeviceAccurateSumSmemAtomic;
    DeviceAccurateSumSmemAtomic().AddTail(d_tail, tail_len, d_bins);
}

enum AccurateFPSumAlgorithm {
    ACCUSUM_SORT_REDUCE = 0,        //< sum binning with sort-reduce
    ACCUSUM_SMEM_ATOMIC = 1,        //< sum binning with atomic smem
};

struct DeviceAccurateFPSum
{
    struct DefaultSetup
    {
        enum {
            Method                  = ACCUSUM_SORT_REDUCE,
            WarpsPerBlock           = 4,
            ItemsPerThread          = 5,
            Expansions              = 2,
            RadixBits               = 3,
            MinConcurrentBlocks     = 9,
        };
    };

    template <
        int BLOCK_THREADS,
        int ITEMS_PER_THREAD,
        int EXPANSIONS,
        int RADIX_BITS,
        int MIN_CONCURRENT_BLOCKS>
    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t SumSortReduce
    (
        double         *d_in,
        int             num_items,
        double          *d_out,
        void           *d_temp_storage,
        void           *h_temp_storage,
        size_t          &temp_storage_bytes,
        cudaStream_t    stream                  = 0)
    {
        typedef AccumulatorBinsMetadata<BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS> BinMeta;
        typedef cub::TexObjInputIterator<double> InputIteratorTexture;

        void *d_bin_sets = NULL;
        void *d_global_bin_set = NULL;
        ExtremeFlags *d_extreme_flags = NULL;
        void *h_bin_sets = NULL;
        void *h_global_bin_set = NULL;
        ExtremeFlags *h_extreme_flags = NULL;
        cudaError_t error = cudaSuccess;

        do {
            int device_id, sm_count, max_blocks_per_sm;
            if (error = CubDebug(cudaGetDevice(&device_id))) break;
            if (error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id))) break;
            void (*AccusumKernel)(InputIteratorTexture,int,void*,size_t,void*,size_t,ExtremeFlags*) =
                DeviceAccurateSumKernel<cub::TexObjInputIterator<double>, BLOCK_THREADS, ITEMS_PER_THREAD, EXPANSIONS, RADIX_BITS, MIN_CONCURRENT_BLOCKS>;
            MaxSmOccupancy(
                max_blocks_per_sm, AccusumKernel, BLOCK_THREADS);
            int grid_size = CUB_MAX(sm_count * max_blocks_per_sm, CUB_ROUND_UP_NEAREST(CUB_QUOTIENT_CEILING(num_items, BinMeta::BIN_CAPACITY), sm_count));
            // Temporary storage allocation requirements
            void* allocations[2];
            void* h_allocations[2];
            size_t allocation_sizes[2] =
            {
                grid_size * BinMeta::NUM_BINS * BinMeta::BIN_SIZE_BYTES,      // for the per-block bin sets
                //            sizeof(double) * (EXPANSIONS + 1) * BinMeta::NUM_BINS,     // for mega-bins
                sizeof(ExtremeFlags)                                        // for nan,inf,inf flags
            };
            if (error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;
            if (error = CubDebug(AliasTemporaries(h_temp_storage, temp_storage_bytes, h_allocations, allocation_sizes))) break;

            if (d_temp_storage == NULL || h_temp_storage == NULL)
            {
                // Return if the caller is simply requesting the size of the storage allocation
                return cudaSuccess;
            }

            d_bin_sets          = allocations[0];
            d_global_bin_set    = NULL; //allocations[0];
            d_extreme_flags     = (ExtremeFlags*)allocations[1];
            h_bin_sets          = h_allocations[0];
            h_global_bin_set    = NULL; //allocations[0];
            h_extreme_flags     = (ExtremeFlags*)h_allocations[1];

            if (h_bin_sets == NULL || h_extreme_flags == NULL)
            {
                error = cudaErrorMemoryAllocation;
                break;
            }

            if (error = CubDebug(cudaMemsetAsync(d_temp_storage, 0, temp_storage_bytes, stream))) break;

            typedef cub::TexObjInputIterator<double> InputIteratorTexture;
            InputIteratorTexture itr;
            itr.BindTexture(d_in, sizeof(double) * num_items);

            //warm-up
            AccusumKernel<<<grid_size, BLOCK_THREADS, 0, stream>>>(
            itr/*NULL*/,0,NULL,0,NULL,0,NULL);

            //cudaProfilerStart();
            AccusumKernel<<<grid_size, BLOCK_THREADS, 0, stream>>>(
                itr,//d_in,
                num_items,
                d_bin_sets,
                0,//temp_reduce_size,
                d_global_bin_set,
                0,//temp_global_bin_set_size,
                d_extreme_flags);
            //cudaProfilerStop();

//            double* h_items = (double*)malloc(num_items * sizeof(double));
//            CubDebugExit(cudaMemcpy(h_items, d_in, num_items * sizeof(double), cudaMemcpyDeviceToHost));
//            printf("%g %g %g\n", h_items[0], h_items[1], h_items[2]);
//            free(h_items);

            if (error = CubDebug(cudaMemcpyAsync(h_bin_sets, d_bin_sets, allocation_sizes[0], cudaMemcpyDeviceToHost, stream))) break;
            //        if (error = CubDebug(cudaMemcpy(h_global_bin_set, d_global_bin_set, allocation_sizes[0], cudaMemcpyDeviceToHost, stream))) break;
            if (error = CubDebug(cudaMemcpyAsync(h_extreme_flags, d_extreme_flags, allocation_sizes[1], cudaMemcpyDeviceToHost, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;

            itr.UnbindTexture();

            double result;
            if (h_extreme_flags->nan)
            {
                result = 0.0 / 0.0;  //NaN
            }
            else
            {
                AccumulatorDouble<EXPANSIONS+1> total_sum(0.0);
                for (int i = 0; i < allocation_sizes[0] / sizeof(double); i++)
                {
                    total_sum.Add(((double*)h_bin_sets)[i]);
//                    if (fabs(((double*)h_bin_sets)[i]) > 0.0 )
//                        printf("%f\n", ((double*)h_bin_sets)[i]);
//                    total_sum.print(); printf("\n");
                }
                total_sum.Normalize();
                result = total_sum[0];
            }

            if (error = CubDebug(cudaMemcpyAsync(d_out, &result, sizeof(double), cudaMemcpyHostToDevice, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;
        } while(0);

//        if(h_bin_sets != NULL)          free(h_bin_sets);
//        if(h_global_bin_set != NULL)    free(h_global_bin_set);
//        if(h_extreme_flags != NULL)     free(h_extreme_flags);

        return error;
    }

    template<int NUM_BINS>
    struct BinidOperator
    {
        __host__ __device__ __forceinline__
        BinidOperator() {}
        __device__ __forceinline__
        int operator()(const double& d) const { return binid(d); }

        __device__ __forceinline__
        int binid(double d) const
        {
            enum {
                LOG_NUM_BINS = Log2<NUM_BINS>::VALUE,
                SHIFT_RIGHT = 63 - LOG_NUM_BINS
            };
            long long ll = __double_as_longlong(d);
            int bin = (int)(ll >> SHIFT_RIGHT) & (NUM_BINS-1);
            return bin;
        }

        double* d_in;
    };

    CUB_RUNTIME_FUNCTION __forceinline__
    static cudaError_t SumSmemAtomic
    (
        double         *d_in,
        int            &num_items,
        double         *d_out,
        void           *d_temp_storage,
        size_t         &temp_storage_bytes,
        cudaStream_t   stream                  = 0)
    {
        enum
        {
            BLOCK_WAVE_SIZE         = 3,        //< number of blocks that run on an SM concurrently
            NUM_BLOCK_WAVES         = 8,        //< multiplication factor for number of blocks
            BLOCK_THREADS           = 672,
            NUM_BIN_COPIES_SMEM     = 1,//15,       //< number of set of bins in shared memory
            NUM_BINS                = 64,
            EXPANSIONS              = 2,
        };

        /**
         * NOTE: The parameter NUM_BLOCK_WAVES needs to be tuned.
         * Increasing NUM_BLOCK_WAVES => more efficient utilization of the SM, but more work for CPU
         */
        void* d_bin_sets = NULL;
        void* h_bin_sets = NULL;
        cudaError_t error = cudaSuccess;

        do {

            int device_id, sm_count;
            if (error = CubDebug(cudaGetDevice(&device_id))) break;
            if (error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id))) break;

            // A wave of blocks is a set of blocks that execute at the same time on all SMs.
            // The number of blocks in a wave is #SMs x number of blocks that fit in an SM.
            // An integer number of
            int grid_size = sm_count * BLOCK_WAVE_SIZE * NUM_BLOCK_WAVES;

//            printf("fixing sm_count to 1\n");
//            printf("fixing grid size to 1\n");
//            sm_count = 1;
//            grid_size = 1;


            int num_tail_items = num_items % (BLOCK_THREADS * grid_size);
            int num_kernel_items = CUB_ROUND_DOWN_NEAREST(num_items, BLOCK_THREADS * grid_size);

//            if (num_tail_items > 0)
//            {
//                printf("[SumSmemAtomic] The smem-atomic accurate summation method currently only supports arrays whose size is a multiple of %d\n", (BLOCK_THREADS * grid_size));
//                printf("[SumSmemAtomic] Working on %d/%d items.\n", num_kernel_items ,num_items);
//                num_items = num_kernel_items;
//            }

            ////
//            cub::TexObjInputIterator<unsigned char> d_samples_tex_itr;
            //d_samples_tex_itr.BindTexture(d_samples, num_items * sizeof(unsigned char));
            typedef BinidOperator<NUM_BINS> BinidOperator;
            typedef TransformInputIterator<int, BinidOperator, double*> BinidIterator;
            BinidIterator binid_iterator(d_in, BinidOperator() );

//            // Determine temporary device storage requirements
//            size_t histogram_bytes = NUM_BINS * sizeof(int);
//            size_t temp_storage_bytes_histogram = 0;
//            DeviceHistogram::HistogramEven<BinidIterator,int,int,int>
//            (   NULL, temp_storage_bytes_histogram, binid_iterator, NULL,
//                NUM_BINS+1, 0, NUM_BINS+1,
//                num_items,stream);
////


            // Temporary storage allocation requirements
            void* allocations[3];
            size_t allocation_sizes[3] =
            {
                grid_size * NUM_BIN_COPIES_SMEM * (EXPANSIONS * NUM_BINS) * sizeof(double),  // bytes needed for bin sets
//                histogram_bytes,
//                temp_storage_bytes_histogram,
            };

            // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
            if (CubDebug(error = AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes))) break;

            // Return if the caller is simply requesting the size of the storage allocation
            if (d_temp_storage == NULL)
                return cudaSuccess;

//            int* d_histogram = (int*) allocations[1];
//            void* d_temp_storage_histogram = allocations[2];
//            int* h_histogram = (int*)malloc(histogram_bytes);

            h_bin_sets = malloc(temp_storage_bytes);
            d_bin_sets = d_temp_storage;
            if (h_bin_sets == NULL)
            {
                error = cudaErrorMemoryAllocation;
                break;
            }
            if (error = CubDebug(cudaMemsetAsync(d_temp_storage, 0, temp_storage_bytes, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;

//            DeviceHistogram::HistogramEven<BinidIterator,int,int,int>
//            (   d_temp_storage_histogram, temp_storage_bytes_histogram, binid_iterator, d_histogram,
//                NUM_BINS+1, 0, NUM_BINS+1,
//                num_items,stream);
//            if (error = CubDebug(cudaMemcpyAsync(h_histogram, d_histogram, histogram_bytes, cudaMemcpyDeviceToHost, stream))) break;
//            if (error = CubDebug(cudaStreamSynchronize(stream))) break;

//            printf("Histogram:\n");
//            for(int i = 0; i < NUM_BINS; i++)
//            {
//                printf("%5d ", h_histogram[i]);
//            }


            cudaProfilerStart();
            // Run kernel once to prime caches and check result
            atomicadd_kernel<
                BLOCK_THREADS,
                BLOCK_WAVE_SIZE,
                NUM_BINS,
                NUM_BIN_COPIES_SMEM>
            <<<grid_size, BLOCK_THREADS, 0, stream>>>(
                d_in,
                num_kernel_items,
                (double2*)d_bin_sets);

            atomicadd_tail_kernel<
                BLOCK_THREADS,
                NUM_BINS>
            <<<1, BLOCK_THREADS, 0, stream>>>(
                &d_in[num_kernel_items],
                num_tail_items,
                (double2*)d_bin_sets);

            if (error = CubDebug(cudaMemcpyAsync(h_bin_sets, d_bin_sets, temp_storage_bytes, cudaMemcpyDeviceToHost, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;

            // Sum all the bins to get final result
            // Summation order is from smaller to larger bin words
            double result;
            AccumulatorDouble<EXPANSIONS+1> total_sum(0.0);

            for (int ibin = 0; ibin < NUM_BINS; ibin++)
            {
                for (int iword = 0; iword < EXPANSIONS; iword++)
                {
                    for (int ibinset = 0; ibinset < grid_size; ibinset++)
                    {
                        for (int ibincopy = 0; ibincopy < NUM_BIN_COPIES_SMEM; ibincopy++)
                        {
                            int idx =
                                ibinset * NUM_BIN_COPIES_SMEM * NUM_BINS * EXPANSIONS +
                                ibincopy * NUM_BINS * EXPANSIONS +
                                ibin * EXPANSIONS +
                                iword;
                            total_sum.Add(((double*)h_bin_sets)[idx]);
                        }
                    }
                }
            }
//            for (int i = 0; i < temp_storage_bytes / sizeof(double); i++)
//            {
//                total_sum.Add(((double*)h_bin_sets)[i]);
//                total_sum.print();
//                printf("\n");
//            }
            total_sum.Normalize();
            result = total_sum[0];
            if (error = CubDebug(cudaMemcpyAsync(d_out, &result, sizeof(double), cudaMemcpyHostToDevice, stream))) break;
            if (error = CubDebug(cudaStreamSynchronize(stream))) break;
            cudaProfilerStop();
        } while(0);

        if (h_bin_sets) free(h_bin_sets);
        return error;
    }

    static cudaError_t Sum(
        double         *d_in,
        int             num_items,
        double         *d_out,
        void           *d_temp_storage,
        void           *h_temp_storage,
        size_t          &temp_storage_bytes,
        cudaStream_t    stream                  = 0
        )
    {

        enum {
            BLOCK_THREADS   = CUB_PTX_WARP_THREADS * DefaultSetup::WarpsPerBlock,
        };
        if (DefaultSetup::Method == (int)ACCUSUM_SORT_REDUCE)
        {
            return SumSortReduce<
                BLOCK_THREADS,
                DefaultSetup::ItemsPerThread,
                DefaultSetup::Expansions,
                DefaultSetup::RadixBits,
                DefaultSetup::MinConcurrentBlocks
            >
            (d_in, num_items, d_out, d_temp_storage, h_temp_storage, temp_storage_bytes, stream);
        }
        else    //ACCUSUM_SMEM_ATOMIC
        {
            return SumSmemAtomic(d_in, num_items, d_out, d_temp_storage, temp_storage_bytes, stream);
        }
    }
};

}               // CUB namespace
CUB_NS_POSTFIX  // Optional outer namespace(s)
