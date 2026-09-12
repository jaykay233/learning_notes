// RUN: lab1-opt %s --pass-pipeline='builtin.module(func.func(lab1-eliminate-identity))' | FileCheck %s
// 负例：没有 identity，图结构应保持（仍有 addf）

module {
  func.func @no_id(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %sum = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %sum : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @no_id
// CHECK: arith.addf
// CHECK-NOT: lab.identity
