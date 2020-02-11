#pragma once

#include <cutlass/cutlass.h>

#include <cutlass/aligned_buffer.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/layout/vector.h>
#include <cutlass/numeric_types.h>

#include <cutlass/core_io.h>

#include <cutlass/gemm/threadblock/default_mma_core_simt.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm75.h>
#include <cutlass/gemm/threadblock/default_mma_core_sm70.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator.h>
#include <cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h>
#include <cutlass/cutlass.h>
#include <cutlass/platform/platform.h>

#include <cutlass/arch/wmma.h>
#include <cutlass/gemm/threadblock/default_mma_core_wmma.h>

#define USE_MMA 1

template <class ElementA, class LayoutA, class ElementB, class LayoutB, class ElementC, class LayoutC,
          class ThreadblockShape, class WarpShape>
__device__ void mma_cutlass_threadblock(ElementA *smem_a, ElementB *smem_b, ElementC *smem_c)
{

#if USE_MMA
  // static const int kStages = 2;
  // Define the MmaCore components
  using MmaCore =
    typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, cutlass::gemm::GemmShape<8, 8, 4>,
                                                        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                                                        cutlass::arch::OpClassTensorOp>;
#else
  static const int kStages = 1;
  // Define the MmaCore components
  using MmaCore =
    typename cutlass::gemm::threadblock::DefaultMmaCore<ThreadblockShape, WarpShape, cutlass::gemm::GemmShape<16, 16, 16>,
                                                        ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC,
                                                        cutlass::arch::OpClassWmmaTensorOp, kStages>;
#endif

  // Define iterators over tiles from the A operand
  using IteratorA
    = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
                                                              ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

  // Define iterators over tiles from the B operand
  using IteratorB
    = cutlass::transform::threadblock::PredicatedTileIterator<cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
                                                              ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

#if USE_MMA
  // Define MmaPipeline Single Stage
  using Mma
    = cutlass::gemm::threadblock::MmaSingleStage<typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
                                                 IteratorB, typename MmaCore::SmemIteratorB, ElementC, LayoutC,
                                                 typename MmaCore::MmaPolicy>;
#else
  using Mma
    = cutlass::gemm::threadblock::MmaPipelined<typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
                                               IteratorB, typename MmaCore::SmemIteratorB, ElementC, LayoutC,
                                               typename MmaCore::MmaPolicy>;
#endif

  // __shared__ typename Mma::SharedStorage shared_storage;
  typename Mma::SharedStorage *shared_storage = reinterpret_cast<typename Mma::SharedStorage *>(smem_a);
  // printf("SharedStorage = %d\n", sizeof(Mma::SharedStorage));
  
  // Compute position within threadblock
  int tb_thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  typename Mma::IteratorA::Layout layout_A = Mma::IteratorA::Layout::packed({Mma::Shape::kM, Mma::Shape::kK});
  typename Mma::IteratorB::Layout layout_B = Mma::IteratorB::Layout::packed({Mma::Shape::kK, Mma::Shape::kN});

  typename Mma::IteratorA::Params params_A(layout_A);
  typename Mma::IteratorB::Params params_B(layout_B);

  typename Mma::IteratorA::Pointer ptr_A = smem_a;
  typename Mma::IteratorB::Pointer ptr_B = smem_b;

  // Construct iterators to A and B operands
  typename Mma::IteratorA iterator_A(params_A, ptr_A, {Mma::Shape::kM, Mma::Shape::kK}, tb_thread_id); // offset = 0
  typename Mma::IteratorB iterator_B(params_B, ptr_B, {Mma::Shape::kK, Mma::Shape::kN}, tb_thread_id); // offset = 0

  constexpr int warp_size = 32;
  int warp_id = tb_thread_id / warp_size;
  int lane_id = tb_thread_id % warp_size;

  // Construct thread-scoped matrix multiply
  Mma mma(*shared_storage, tb_thread_id, warp_id, lane_id);

  typename Mma::FragmentC accum;

  accum.clear();

  int gemm_k_iterations = (Mma::Shape::kK + Mma::Shape::kK - 1) / Mma::Shape::kK;

  // Compute threadblock-scoped matrix multiply-add
  mma(gemm_k_iterations, accum, iterator_A, iterator_B, accum);

  // Output results
  typename Mma::Operator::IteratorC iterator_C({smem_c, Mma::Shape::kN}, lane_id);

  iterator_C.add_tile_offset({warp_id % Mma::WarpCount::kM, warp_id / Mma::WarpCount::kM});

  __syncthreads();

  iterator_C.store(accum);
}

#undef USE_MMA

