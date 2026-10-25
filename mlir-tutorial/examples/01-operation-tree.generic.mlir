"builtin.module"() ({
  "func.func"() <{function_type = (i1, i32, i32) -> i32, sym_name = "choose"}> ({
  ^bb0(%arg0: i1, %arg1: i32, %arg2: i32):
    %0 = "scf.if"(%arg0) ({
      "scf.yield"(%arg1) : (i32) -> ()
    }, {
      "scf.yield"(%arg2) : (i32) -> ()
    }) : (i1) -> i32
    "func.return"(%0) : (i32) -> ()
  }) : () -> ()
}) : () -> ()
