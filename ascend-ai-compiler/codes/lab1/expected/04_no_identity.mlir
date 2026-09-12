// 期望：无 identity 可消，与输入等价
module {
  func.func @no_id(%arg0: tensor<4xf32>, %arg1: tensor<4xf32>) -> tensor<4xf32> {
    %sum = arith.addf %arg0, %arg1 : tensor<4xf32>
    return %sum : tensor<4xf32>
  }
}
