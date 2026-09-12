#include "lab1/LabDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::lab;

LabDialect::LabDialect(MLIRContext *ctx) : Dialect(getDialectNamespace(), ctx, TypeID::get<LabDialect>()) {
  addOperations<IdentityOp>();
}

void mlir::lab::registerLabDialect(DialectRegistry &registry) {
  registry.insert<LabDialect>();
}

void IdentityOp::build(OpBuilder &builder, OperationState &state, Value input) {
  state.addOperands(input);
  state.addTypes(input.getType());
}

ParseResult IdentityOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand input;
  Type type;
  if (parser.parseOperand(input) || parser.parseColonType(type))
    return failure();
  if (parser.resolveOperand(input, type, result.operands))
    return failure();
  result.addTypes(type);
  return success();
}

void IdentityOp::print(OpAsmPrinter &printer) {
  printer << " " << getInput() << " : "
          << getOperation()->getResult(0).getType();
}
