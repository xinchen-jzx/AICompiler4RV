// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/InliningUtils.h"

#include "RISCV/RISCVDialect.h"
#include "RISCV/RISCVOps.h"

using namespace mlir;
using namespace riscv;   

//===----------------------------------------------------------------------===//
// RISCV dialect.
//===----------------------------------------------------------------------===//

#include "RISCV/RISCVOpsDialect.cpp.inc"
// #include "RISCV/RISCVDialect.cpp.inc"


void RISCVDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "RISCV/RISCVOps.cpp.inc"
      >();
}
//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//
void riscv::ConstantOp::build(mlir::OpBuilder &builder,
                              mlir::OperationState &state, double value) {
  auto dataType = RankedTensorType::get({}, builder.getF64Type());
  auto dataAttribute = DenseElementsAttr::get(dataType, value);
  riscv::ConstantOp::build(builder, state, dataType, dataAttribute);
}
//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//
void TransposeOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        mlir::Value value) {
  state.addTypes(UnrankedTensorType::get(builder.getF64Type()));
  state.addOperands(value);
}

// void TransposeOp::inferShapes() {
//   auto arrayTy = llvm::cast<RankedTensorType>(getOperand().getType());
//   SmallVector<int64_t, 2> dims(llvm::reverse(arrayTy.getShape()));
//   getResult().setType(RankedTensorType::get(dims, arrayTy.getElementType()));
// }

// mlir::LogicalResult TransposeOp::verify() {
//   auto inputType = llvm::dyn_cast<RankedTensorType>(getOperand().getType());
//   auto resultType = llvm::dyn_cast<RankedTensorType>(getType());
//   if (!inputType || !resultType)
//     return mlir::success();

//   auto inputShape = inputType.getShape();
//   if (!std::equal(inputShape.begin(), inputShape.end(),
//                   resultType.getShape().rbegin())) {
//     return emitError()
//            << "expected result shape to be a transpose of the input";
//   }
//   return mlir::success();
// }
//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//
// CallInterfaceCallable GenericCallOp::getCallableForCallee() {
//   return (*this)->getAttrOfType<SymbolRefAttr>("callee");
// }

// void GenericCallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
//   (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
// }

// void FuncOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
//                    llvm::StringRef name, mlir::FunctionType type,
//                    llvm::ArrayRef<mlir::NamedAttribute> attrs) {
//   // FunctionOpInterface provides a convenient `build` method that will populate
//   // the state of our FuncOp, and create an entry block.
//   buildWithEntryBlock(builder, state, name, type, attrs, type.getInputs());
// }

// mlir::ParseResult FuncOp::parse(mlir::OpAsmParser &parser,
//                                 mlir::OperationState &result) {
//   // Dispatch to the FunctionOpInterface provided utility method that parses the
//   // function operation.
//   auto buildFuncType =
//       [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
//          llvm::ArrayRef<mlir::Type> results,
//          mlir::function_interface_impl::VariadicFlag,
//          std::string &) { return builder.getFunctionType(argTypes, results); };

//   return mlir::function_interface_impl::parseFunctionOp(
//       parser, result, /*allowVariadic=*/false,
//       getFunctionTypeAttrName(result.name), buildFuncType,
//       getArgAttrsAttrName(result.name), getResAttrsAttrName(result.name));
// }

// void FuncOp::print(mlir::OpAsmPrinter &p) {
//   // Dispatch to the FunctionOpInterface provided utility method that prints the
//   // function operation.
//   mlir::function_interface_impl::printFunctionOp(
//       p, *this, /*isVariadic=*/false, getFunctionTypeAttrName(),
//       getArgAttrsAttrName(), getResAttrsAttrName());
// }


mlir::Operation *RISCVDialect::materializeConstant(mlir::OpBuilder &builder,
                                                   mlir::Attribute value,
                                                   mlir::Type type,
                                                   mlir::Location loc) {
  return builder.create<riscv::ConstantOp>(
      loc, type, mlir::cast<mlir::DenseElementsAttr>(value));
}