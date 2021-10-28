# add doxygen add doxygen documentation note that cmake 3.9 introduced a nicer way to do this but we don't want to
# require cmake 3.9 by default yet

option(QUDA_GENERATE_DOXYGEN "generate doxygen documentation")

if(QUDA_GENERATE_DOXYGEN)
  find_package(Doxygen)

  if(DOXYGEN_FOUND)
    if(DOXYGEN_DOT_FOUND)
      get_filename_component(DOXYGEN_DOT_PATH ${DOXYGEN_DOT_EXECUTABLE} DIRECTORY)
    endif()
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)
    configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${DOXYGEN_OUT} @ONLY)

    add_custom_target(
      doc
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      COMMENT "Generating doxygen documentation"
      VERBATIM)
  endif()
endif()
