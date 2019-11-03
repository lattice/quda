HIP_PATH?= $(wildcard /opt/rocm/hip)
ifeq (,$(HIP_PATH))
	HIP_PATH=../../..
endif
HIPCC=$(HIP_PATH)/bin/hipcc
THRUST_PATH=..

#### GCC system includes workaround ####
GCC_VER ?= 4.8

GCC_CUR_VER = $(shell gcc -dumpversion)
GPP_CUR_VER = $(shell g++ -dumpversion)

GCC_CUR = 0
GPP_CUR = 1

ifeq ($(findstring $(GCC_VER),$(GCC_CUR_VER)),$(GCC_VER))
GCC_CUR = GCC_VER
endif

ifeq ($(findstring $(GCC_VER),$(GPP_CUR_VER)),$(GCC_VER))
GPP_CUR = GCC_VER
endif

ifeq ($(GCC_CUR), $(GPP_CUR))
    CXXFLAGS += -I /usr/include/x86_64-linux-gnu -I /usr/include/x86_64-linux-gnu/c++/$(GCC_VER) -I /usr/include/c++/$(GCC_VER) -I $(THRUST_PATH) -I ./cuda -I ./cpp_integration
else
    $(warning )
    $(warning ***************************************************)
    $(warning *** The supported version of gcc and g++ is $(GCC_VER) ***)
    $(warning ***    Current default version of gcc is $(GCC_CUR_VER)    ***)
    $(warning ***    Current default version of g++ is $(GPP_CUR_VER)    ***)
    $(warning ***************************************************)
    $(warning )
endif
#### GCC system includes workaround ####

ifeq ($(HIP_PLATFORM), nvcc)
CXXFLAGS += -std=c++11 -O3 -DUSE_TEXTURES -D__NVCC__ -D __HIPCC__
else
CXXFLAGS += -std=c++11 -O3 -D __HIPCC__
endif

all: arbitrary_transformation basic_vector bounding_box bucket_sort2d constant_iterator counting_iterator device_ptr discrete_voronoi dot_products_with_zip expand fill_copy_sequence histogram lambda lexicographical_sort max_abs_diff minimal_custom_backend minmax mode monte_carlo monte_carlo_disjoint_sequences norm padded_grid_reduction permutation_iterator raw_reference_cast remove_points2d repeated_range run_length_decoding run_length_encoding saxpy scan_by_key set_operations simple_moving_average sort sorting_aos_vs_soa sparse_vector stream_compaction strided_range sum summary_statistics summed_area_table sum_rows tiled_range transform_iterator transform_output_iterator uninitialized_vector version weld_vertices word_count async_reduce custom_temporary_allocation fallback_allocator range_view simple_cuda_streams unwrap_pointer wrap_pointer device host
 
arbitrary_transformation: arbitrary_transformation.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

basic_vector: basic_vector.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

bounding_box: bounding_box.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

bucket_sort2d: bucket_sort2d.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

constant_iterator: constant_iterator.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

counting_iterator: counting_iterator.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

device_ptr: device_ptr.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

discrete_voronoi: discrete_voronoi.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

dot_products_with_zip: dot_products_with_zip.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

expand: expand.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

fill_copy_sequence: fill_copy_sequence.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

histogram: histogram.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

lambda: lambda.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

lexicographical_sort: lexicographical_sort.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

max_abs_diff: max_abs_diff.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

minimal_custom_backend: minimal_custom_backend.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

minmax: minmax.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

mode: mode.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

monte_carlo: monte_carlo.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

monte_carlo_disjoint_sequences: monte_carlo_disjoint_sequences.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

norm: norm.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

padded_grid_reduction: padded_grid_reduction.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

permutation_iterator: permutation_iterator.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

raw_reference_cast: raw_reference_cast.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

remove_points2d: remove_points2d.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

repeated_range: repeated_range.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

run_length_decoding: run_length_decoding.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

run_length_encoding: run_length_encoding.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

saxpy: saxpy.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

scan_by_key: scan_by_key.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

set_operations: set_operations.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

simple_moving_average: simple_moving_average.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

sort: sort.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

sorting_aos_vs_soa: sorting_aos_vs_soa.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

sparse_vector: sparse_vector.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

stream_compaction: stream_compaction.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

strided_range: strided_range.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

sum: sum.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

summary_statistics: summary_statistics.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

summed_area_table: summed_area_table.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

sum_rows: sum_rows.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

tiled_range: tiled_range.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

transform_iterator: transform_iterator.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

transform_output_iterator: transform_output_iterator.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

uninitialized_vector: uninitialized_vector.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

version: version.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

weld_vertices: weld_vertices.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

word_count: word_count.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

async_reduce: cuda/async_reduce.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

custom_temporary_allocation: cuda/custom_temporary_allocation.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

fallback_allocator: cuda/fallback_allocator.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

range_view: cuda/range_view.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

simple_cuda_streams: cuda/simple_cuda_streams.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

unwrap_pointer: cuda/unwrap_pointer.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

wrap_pointer: cuda/wrap_pointer.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

device: cpp_integration/device.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

host: cpp_integration/host.cu
ifeq ($(shell which $(HIPCC) > /dev/null; echo $$?), 0)
	${HIPCC} ${CXXFLAGS} -o $@ $^
else
	$(error "Cannot find $(HIPCC), please install HIP toolkit")
endif

.PHONY: clean

clean:
	rm -f arbitrary_transformation basic_vector bounding_box bucket_sort2d constant_iterator counting_iterator device_ptr discrete_voronoi dot_products_with_zip expand fill_copy_sequence histogram lambda lexicographical_sort max_abs_diff minimal_custom_backend minmax mode monte_carlo monte_carlo_disjoint_sequences norm padded_grid_reduction permutation_iterator raw_reference_cast remove_points2d repeated_range run_length_decoding run_length_encoding saxpy scan_by_key set_operations simple_moving_average sort sorting_aos_vs_soa sparse_vector stream_compaction strided_range sum summary_statistics summed_area_table sum_rows tiled_range transform_iterator transform_output_iterator uninitialized_vector version weld_vertices word_count async_reduce custom_temporary_allocation fallback_allocator range_view simple_cuda_streams unwrap_pointer wrap_pointer device host *.o

