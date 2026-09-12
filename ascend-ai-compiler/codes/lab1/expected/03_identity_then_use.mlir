// 期望：identity 消去后，addf 直接使用 %arg0/%arg1
module {
  func.func @then_use(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %sum = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %sum : tensor<4xf32>
  }
}
