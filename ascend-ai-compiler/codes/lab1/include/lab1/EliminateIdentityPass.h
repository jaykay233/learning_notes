#ifndef LAB1_ELIMINATEIDENTITYPASS_H
#define LAB1_ELIMINATEIDENTITYPASS_H

#include <memory>

namespace mlir {
class Pass;
namespace lab {
/// Create the pass: eliminate lab.identity via greedy patterns.
std::unique_ptr<Pass> createEliminateIdentityPass();
void registerEliminateIdentityPass();
} // namespace lab
} // namespace mlir

#endif
