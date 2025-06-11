add_executable(multihost_unit_tests)
target_sources(multihost_unit_tests PRIVATE test_multihost_control_plane.cpp)

target_link_libraries(
    multihost_unit_tests
    PRIVATE
        tt_metal
        test_common_libs
)

target_include_directories(
    multihost_unit_tests
    PRIVATE
        "$<TARGET_PROPERTY:Metalium::Metal,INCLUDE_DIRECTORIES>"
        ${PROJECT_SOURCE_DIR}/tests
)

set_target_properties(
    multihost_unit_tests
    PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY
            ${PROJECT_BINARY_DIR}/test/tt_metal/distributed/multihost
)
