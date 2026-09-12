// RUN: lab1-opt %s --pass-pipeline='builtin.module(func.func(lab1-eliminate-identity))' | FileCheck %s
// identity 后再参与计算：消去后 addi 应直接用 %arg0 / %arg1

module {
  func.func @then_use(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %x = lab.identity %arg0 : tensor<4xf32>
    %y = lab.identity %arg1 : tensor<4xf32>
    %sum = arith.addf %x, %y : tensor<4xf32>
    return %sum : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @then_use
// CHECK-NOT: lab.identity
// CHECK: %[[SUM:.*]] = arith.addf %arg0, %arg1
// CHECK: return %[[SUM]]
