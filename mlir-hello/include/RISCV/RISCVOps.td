#ifndef RISCV_OPS
#define RISCV_OPS

include "RISCVDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def RISCV_FMAOp : RISCV_Op<"fma", [Pure]> {
  let summary = "fused multiply-add";
  let description = [{
    Fused multiply-add operation computes (a * b) + c with a single rounding.
    For example:

    ```mlir
      %0 = "riscv.fma"(%a, %b, %c) : (f32, f32, f32) -> f32
    ```
  }];

  let arguments = (ins F32:$a, F32:$b, F32:$c);
  let results = (outs F32);
}