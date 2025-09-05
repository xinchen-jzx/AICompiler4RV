configure_lit_site_cfg(
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
        ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
        MAIN_CONFIG
        ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(RISCV_TEST_DEPENDS
        FileCheck count not
        mlir-opt
        riscv-opt
        )

add_lit_testsuite(check-riscv "Running the riscv regression tests"
        ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${RISCV_TEST_DEPENDS}
        )
set_target_properties(check-riscv PROPERTIES FOLDER "Tests")

add_lit_testsuites(RISCV ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${RISCV_TEST_DEPENDS})
