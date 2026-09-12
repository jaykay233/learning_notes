//===----------------------------------------------------------------------===//
// Simplified hand-written dialect for Lab1 (no TableGen).
// API details may need tiny tweaks for your LLVM version.
//===----------------------------------------------------------------------===//

#ifndef LAB1_LABDIALECT_H
#define LAB1_LABDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir {
namespace lab {

class LabDialect : public Dialect {
public:
  explicit LabDialect(MLIRContext *ctx);
  static StringRef getDialectNamespace() { return "lab"; }
};

/// %y = lab.identity %x : type
class IdentityOp : public Op<IdentityOp, OpTrait::OneResult, OpTrait::OneOperand,
                             OpTrait::SameOperandsAndResultType> {
public:
  using Op::Op;

  static ArrayRef<StringRef> getAttributeNames() { return {}; }
  static StringRef getOperationName() { return "lab.identity"; }

  static void build(OpBuilder &builder, OperationState &state, Value input);
  static ParseResult parse(OpAsmParser &parser, OperationState &result);
  void print(OpAsmPrinter &printer);

  Value getInput() { return getOperand(); }
};

void registerLabDialect(DialectRegistry &registry);

} // namespace lab
} // namespace mlir

#endif
