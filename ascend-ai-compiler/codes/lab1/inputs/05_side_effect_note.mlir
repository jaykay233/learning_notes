# Fix Affine in 05 - use memref.store which needs memref dialect; simplify to comment-only note without affine

module {
  func.func @doc_only() {
    // 有副作用的例子请在报告里讨论，例如：
    //   memref.store / 带副作用的自定义 op
    // 不能因为「长得像单入单出」就当成 lab.identity 删掉。
    return
  }
}
