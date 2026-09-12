#include "lab1/EliminateIdentityPass.h"
#include "lab1/LabDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::arith::ArithDialect,
                  mlir::lab::LabDialect>();

  mlir::lab::registerEliminateIdentityPass();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Lab1 opt\n", registry));
}
