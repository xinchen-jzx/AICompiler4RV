#ifndef RISCV_DIALECT
#define RISCV_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// RISCV dialect definition.
//===----------------------------------------------------------------------===//

def RISCV_Dialect : Dialect {
    // 定义名字空间 namespace，对应 C++ 的 getDialectNamespace 方法返回值
    let name = "riscv";

    // 一行关于这个 Dialect 的介绍
    let summary = "RISC-V dialect.";

    // 更详细的关于这个 Dialect 的介绍
    let description = [{
        This dialect is a minimal example to implement RISC-V operations for MLIR.
    }];

    // 产生一个返回名字空间名称的借口
    let cppNamespace = "::riscv";

    // 该设置用于激活 materializeConstant 方法，这使得可以例如 Canonicalize 优化
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base hello operation definition.
//===----------------------------------------------------------------------===//
// 定义一个 Op 作为后续其他具体 Op 的基类
class RISCV_Op<string mnemonic, list<Trait> traits = []> :
        Op<RISCV_Dialect, mnemonic, traits>;

#endif // RISCV_DIALECT