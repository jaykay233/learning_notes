#include "lab1/EliminateIdentityPass.h"
#include "lab1/LabDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::lab;

namespace {

struct EliminateIdentityPattern : public OpRewritePattern<IdentityOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(IdentityOp op,
                                PatternRewriter &rewriter) const override {
    // ========== BEGIN: Lab1 填写（学生作业核心）==========
    // 语义：%y = lab.identity %x  →  所有用 %y 的地方改成 %x，再删掉 identity
    Value in = op.getInput();
    Value out = op.getResult();
    rewriter.replaceAllUsesWith(out, in);
    rewriter.eraseOp(op);
    return success();
    // ========== END: Lab1 填写 ==========
  }
};

struct EliminateIdentityPass
    : public PassWrapper<EliminateIdentityPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(EliminateIdentityPass)

  StringRef getArgument() const final { return "lab1-eliminate-identity"; }
  StringRef getDescription() const final {
    return "Eliminate lab.identity ops (Lab1)";
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<EliminateIdentityPattern>(&getContext());
    // Greedy：链式 identity 会反复匹配直到消完
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::lab::createEliminateIdentityPass() {
  return std::make_unique<EliminateIdentityPass>();
}

void mlir::lab::registerEliminateIdentityPass() {
  PassRegistration<EliminateIdentityPass>();
}
