option(GINKGO_BUILD_EXTLIB_EXAMPLE "Build the external-lib-interfacing with deal.II, you need to link the deal.II library." OFF)
option(GINKGO_RUN_EXAMPLES "Run the examples." OFF)

function(setup_example_target target_name)
    add_executable(${target_name} ${target_name}.cpp)
    target_link_libraries(${target_name} ginkgo)
    target_include_directories(${target_name} PRIVATE ${PROJECT_SOURCE_DIR})
endfunction()

set(EXAMPLES_EXEC_LIST "adaptiveprecision-blockjacobi;custom-logger;ginkgo-overhead;ginkgo-ranges;ilu-preconditioned-solver;ir-ilu-preconditioned-solver;inverse-iteration;iterative-refinement;mixed-precision-ir;nine-pt-stencil-solver;performance-debugging;poisson-solver;preconditioned-solver;preconditioner-export;simple-solver;simple-solver-logging;three-pt-stencil-solver;")

set(EXAMPLES_LIST "${EXAMPLES_EXEC_LIST};custom-matrix-format;custom-stopping-criterion;minimal-cuda-solver;papi-logging")

if(GINKGO_BUILD_EXTLIB_EXAMPLE)
    set(EXAMPLES_LIST "${EXAMPLES_LIST};external-lib-interfacing")
endif()

foreach(example ${EXAMPLES_LIST})
    add_subdirectory(${example})
endforeach()

set(run_list "")
if(GINKGO_RUN_EXAMPLES)
    foreach(example ${EXAMPLES_LIST})
        add_custom_target("run-${example}"
            COMMAND ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example} > ${CMAKE_CURRENT_BINARY_DIR}/${example}/${example}.out
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${example})
    endforeach()
endif()

set(OUT_FILE "${CMAKE_BINARY_DIR}/example-targets.txt")
file(WRITE ${OUT_FILE} "")
foreach(example ${EXAMPLES_EXEC_LIST})
    file(APPEND ${OUT_FILE} "run-${example}\n")
endforeach()
