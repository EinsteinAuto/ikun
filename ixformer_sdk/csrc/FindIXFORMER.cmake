# use python to find ixformer libs and include
if (CMAKE_VERSION VERSION_LESS 3.18)
    set(DEV_MODULE Development)
else()
    set(DEV_MODULE Development.Module)
endif()

find_package(Python COMPONENTS Interpreter ${DEV_MODULE} REQUIRED)

# find ixformer
set(IXFORMER_FOUND FALSE)

if("${Python_FOUND}" STREQUAL "TRUE")
    execute_process(
            COMMAND ${Python_EXECUTABLE} -c "import os, ixformer; print(os.path.dirname(ixformer.__file__))"
            OUTPUT_VARIABLE IXFORMER_PYDIR
            ERROR_VARIABLE PYTHON_ERROR
            RESULT_VARIABLE PYTHON_RESULT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
    )
    if ("${IXFORMER_PYDIR}" STREQUAL "")
        message("-- Not found ixFormer")
    else ()
        message("-- Found ixFormer: ${IXFORMER_PYDIR}")
        set(IXFORMER_FOUND TRUE)
    endif ()
endif()

set(IXFORMER_COMM_LIBS "ixformer_comm")
set(IXFORMER_KERNEL_LIBS "ixformer_kernels")
set(IXFORMER_LIBS "${IXFORMER_COMM_LIBS} ${IXFORMER_KERNEL_LIBS}")
set(IXFORMER_INCLUDE "")
set(IXFORMER_DIR "")

if("${IXFORMER_FOUND}" STREQUAL "TRUE")
    set(IXFORMER_INCLUDE "${IXFORMER_PYDIR}/csrc/include")
    set(IXFORMER_DIR "${IXFORMER_PYDIR}")
    message("-- ixFormer LIBS: ${IXFORMER_LIBS}, INCLUDE: ${IXFORMER_INCLUDE}")
endif ()
