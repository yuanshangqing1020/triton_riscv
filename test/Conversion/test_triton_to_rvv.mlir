// RUN: triton-opt %s -convert-triton-to-rvv | FileCheck %s

module {
  // CHECK-LABEL: func @kernel
  // CHECK-SAME: (%arg0: memref<?xf32>)
  tt.func @kernel(%arg0: !tt.ptr<f32>) {
    // CHECK: %[[CST1:.*]] = arith.constant dense<1.000000e+00> : vector<128xf32>
    %1 = arith.constant dense<1.0> : tensor<128xf32>
    // CHECK: %[[CST2:.*]] = arith.constant dense<2.000000e+00> : vector<128xf32>
    %2 = arith.constant dense<2.0> : tensor<128xf32>
    // CHECK: arith.addf %[[CST1]], %[[CST2]] : vector<128xf32>
    %3 = arith.addf %1, %2 : tensor<128xf32>
    tt.return
  }
}
