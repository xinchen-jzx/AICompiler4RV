// RUN: riscv-opt %s | FileCheck %s

// CHECK: @riscv_word_string = internal constant [16 x i8] c"RISCV, World! \0A\00"
// CHECK: define void @main()
func.func @main() {
    // CHECK: %{{.*}} = call i32 (ptr, ...) @printf(ptr @riscv_word_string)
    "riscv.world"() : () -> ()
    return
}
