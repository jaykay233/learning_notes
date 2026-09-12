// RUN: lab1-opt %s --pass-pipeline='builtin.module(func.func(lab1-eliminate-identity))' | FileCheck %s
// 链式 identity：应全部消掉，只剩 return %arg0

module {
  func.func @chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    %t0 = lab.identity %arg0 : tensor<4xf32>
    %t1 = lab.identity %t0 : tensor<4xf32>
    return %t1 : tensor<4xf32>
  }
}

// CHECK-LABEL: func.func @chain
// CHECK-NOT: lab.identity
// CHECK: return %arg0
