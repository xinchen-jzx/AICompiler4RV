// RUN: riscv-opt %s | FileCheck %s
// CHECK: define void @main()
func.func @main() {
    %0 = "riscv.constant"() {value = dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "riscv.transpose"(%0) : (tensor<2x3xf64>) -> (tensor<3x2xf64>)
    "riscv.print"(%1) : (tensor<3x2xf64>) -> ()
    return
}

