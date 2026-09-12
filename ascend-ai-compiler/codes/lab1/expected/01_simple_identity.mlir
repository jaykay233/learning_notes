// 期望：identity 已消除，直接 return 参数
module {
  func.func @simple(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
