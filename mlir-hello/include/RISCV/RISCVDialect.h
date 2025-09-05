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

#ifndef HELLO_HELLODIALECT_H
#define HELLO_HELLODIALECT_H

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// #include "RISCV/ShapeInferenceInterface.h"

#include "RISCV/RISCVOpsDialect.h.inc"
// #include "RISCV/RISCVDialect.h.inc"

// namespace mlir {
// namespace riscv {
// namespace detail {
// struct StructTypeStorage;
// } // namespace detail
// } // namespace toy
// } // namespace mlir

// #define GET_OP_CLASSES
// #include "RISCV/RISCVOps.h.inc"

// namespace mlir {
// namespace riscv {

// //===----------------------------------------------------------------------===//
// // Toy Types
// //===----------------------------------------------------------------------===//

// /// This class defines the Toy struct type. It represents a collection of
// /// element types. All derived types in MLIR must inherit from the CRTP class
// /// 'Type::TypeBase'. It takes as template parameters the concrete type
// /// (StructType), the base class to use (Type), and the storage class
// /// (StructTypeStorage).
// class StructType : public mlir::Type::TypeBase<StructType, mlir::Type,
//                                                detail::StructTypeStorage> {
// public:
//   /// Inherit some necessary constructors from 'TypeBase'.
//   using Base::Base;
  
//   static constexpr llvm::StringLiteral name = "riscv.struct";

//   /// Create an instance of a `StructType` with the given element types. There
//   /// *must* be atleast one element type.
//   static StructType get(llvm::ArrayRef<mlir::Type> elementTypes);

//   /// Returns the element types of this struct type.
//   llvm::ArrayRef<mlir::Type> getElementTypes();

//   /// Returns the number of element type held by this struct.
//   size_t getNumElementTypes() { return getElementTypes().size(); }
// };


// } // namespace toy
// } // namespace mlir

#endif // HELLO_HELLODIALECT_H

/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|* From: RISCVDialect.td                                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

// namespace riscv {

// class RISCVDialect : public ::mlir::Dialect {
//   explicit RISCVDialect(::mlir::MLIRContext *context);

//   void initialize();
//   friend class ::mlir::MLIRContext;
// public:
//   ~RISCVDialect() override;
//   static constexpr ::llvm::StringLiteral getDialectNamespace() {
//     return ::llvm::StringLiteral("riscv");
//   }

//   /// Materialize a single constant operation from a given attribute value with
//   /// the desired resultant type.
//   ::mlir::Operation *materializeConstant(::mlir::OpBuilder &builder,
//                                          ::mlir::Attribute value,
//                                          ::mlir::Type type,
//                                          ::mlir::Location loc) override;
// };
// } // namespace riscv
// MLIR_DECLARE_EXPLICIT_TYPE_ID(::riscv::RISCVDialect)
