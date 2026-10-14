# add file I/O sources
target_sources(quda_cpp PRIVATE field_io.cpp gauge_io.cpp milc_common.cpp milc_gauge.cpp vector_io.cpp)

if(QUDA_QIO)
  target_sources(quda_cpp PRIVATE qio_field.cpp layout_hyper.cpp)
endif()
