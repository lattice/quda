# add file I/O sources
target_sources(
  quda_cpp
  PRIVATE field_io.cpp
          gauge_io.cpp
          host_field.cpp
          lime.cpp
          milc_common.cpp
          milc_gauge.cpp
          reunitarize.cpp
          scidac_common.cpp
          scidac_gauge.cpp
          scidac_vector.cpp
          site_io.cpp
          vector_io.cpp)

if(QUDA_QIO)
  target_sources(quda_cpp PRIVATE qio_field.cpp layout_hyper.cpp)
endif()
