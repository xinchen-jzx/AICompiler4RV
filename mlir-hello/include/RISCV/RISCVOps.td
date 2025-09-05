#ifndef RISCV_OPS
#define RISCV_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/VectorInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

include "RISCVDialect.td"
// include "RISCV/ShapeInferenceInterface.td"
def ConstantOp : RISCV_Op<"constant", [Pure]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "riscv.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

//  let parser = [{ return ::parseConstantOp(parser, result); }];
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}

def PrintOp : RISCV_Op<"print", [Pure]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

def TransposeOp : RISCV_Op<"transpose",
    [Pure]> {
    // [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {

  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins "mlir::Value":$input)>
  ];

  // Indicate that additional verification for this operation is necessary.
  // let hasVerifier = 1;
}

// def FuncOp : RISCV_Op<"func", [
//     FunctionOpInterface, IsolatedFromAbove
//   ]> {
//   let summary = "user defined function operation";
//   let description = [{
//     The "riscv.func" operation represents a user defined function. These are
//     callable SSA-region operations that contain riscv computations.

//     Example:

//     ```mlir
//     riscv.func @main() {
//       %0 = riscv.constant dense<5.500000e+00> : tensor<f64>
//       %1 = riscv.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
//       riscv.print %1 : tensor<2x2xf64>
//       riscv.return
//     }
//     ```
//   }];

//   let arguments = (ins
//     SymbolNameAttr:$sym_name,
//     TypeAttrOf<FunctionType>:$function_type,
//     OptionalAttr<DictArrayAttr>:$arg_attrs,
//     OptionalAttr<DictArrayAttr>:$res_attrs
//   );
//   let regions = (region AnyRegion:$body);

//   let builders = [OpBuilder<(ins
//     "StringRef":$name, "FunctionType":$type,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
//   >];
//   let extraClassDeclaration = [{
//     //===------------------------------------------------------------------===//
//     // FunctionOpInterface Methods
//     //===------------------------------------------------------------------===//

//     /// Returns the argument types of this function.
//     ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

//     /// Returns the result types of this function.
//     ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

//     Region *getCallableRegion() { return &getBody(); }
//   }];
//   let hasCustomAssemblyFormat = 1;
//   let skipDefaultBuilders = 1;
// }

def WorldOp : RISCV_Op<"world", [Pure]> {
    let summary = "print RISCV, World";
    let description = [{
        The "world" operation prints "RISCV, World", and produces
        no results.
    }];
}

#endif // RISCV_OPS

// def RISCV_FMAOp : RISCV_Op<"fma", [Pure]> {
//   let summary = "fused multiply-add";
//   let description = [{
//     Fused multiply-add operation computes (a * b) + c with a single rounding.
//     For example:

//     ```mlir
//       %0 = "riscv.fma"(%a, %b, %c) : (f32, f32, f32) -> f32
//     ```
//   }];

//   let arguments = (ins F32:$a, F32:$b, F32:$c);
//   let results = (outs F32);
// }