# Target name
set(target cpp-oss-template)

# Define
set(root_dir ${CMAKE_CURRENT_SOURCE_DIR}/../..)

set(XNNPACK_DIR ${CMAKE_SOURCE_DIR}/Libraries/xnnpack)

# Includes
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../Libraries
    ${XNNPACK_DIR}/include
    ${PTHREADPOOL_SOURCE_DIR}/include
)

message(STATUS ${PTHREADPOOL_SOURCE_DIR}/include)

# Sources
file(GLOB header_dir
    ${root_dir}/Includes)

file(GLOB_RECURSE headers
    ${header_dir}/*.hpp)

file(GLOB_RECURSE sources
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

# Build library
add_library(${target}
    ${sources})

# Project options
set_target_properties(${target}
    PROPERTIES
    ${DEFAULT_PROJECT_OPTIONS}
)

# Compile options
target_compile_options(${target}
    PRIVATE

    PUBLIC
    ${DEFAULT_COMPILE_OPTIONS}

    INTERFACE
)

target_link_libraries(${target}
    PRIVATE
    XNNPACK
    ${operators}

    PUBLIC
    ${DEFAULT_LINKER_OPTIONS}
	${DEFAULT_LIBRARIES}

    INTERFACE
)

target_include_directories(${target} PUBLIC 
${PTHREADPOOL_SOURCE_DIR}/include
)

# Install
install(TARGETS ${target} DESTINATION lib)
install(DIRECTORY ${header_dir} DESTINATION include) 