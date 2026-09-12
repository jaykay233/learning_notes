// RUN: lab1-opt %s --pass-pipeline='builtin.module(func.func(lab1-eliminate-identity))' | FileCheck %s
// 单个 identity：return 应直接使用参数

module {
  func.func @simple(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %y = lab.identity %arg0 : tensor<4xf32>
    return %y : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @simple
// CHECK-NOT: lab.identity
// CHECK: return %arg0
