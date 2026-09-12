// 期望：链式 identity 全部消除
module {
  func.func @chain(%arg0: tensor<4xf32>) -> tensor<4xf32> {
    return %arg0 : tensor<4xf32>
  }
}
