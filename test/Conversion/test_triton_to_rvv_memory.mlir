// RUN: triton-opt %s -convert-triton-to-rvv | FileCheck %s

module {
  // CHECK-LABEL: func @test_load_store
  // CHECK-SAME: (%arg0: memref<?xf32>, %arg1: memref<?xf32>)
  tt.func @test_load_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    // CHECK: %[[OFF:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %off = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
    
    // CHECK: %[[EXT:.*]] = vector.extract %[[OFF]][0] : i32 from vector<4xi32>
    // CHECK: %[[IDX:.*]] = arith.index_cast %[[EXT]] : i32 to index
    // CHECK: %[[VAL:.*]] = vector.transfer_read %arg0[%[[IDX]]], %cst {in_bounds = [true]} : memref<?xf32>, vector<4xf32>
    %ptr0 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %ptr0, %off : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %val = tt.load %ptr : tensor<4x!tt.ptr<f32>>
    
    // CHECK: vector.transfer_write %[[VAL]], %arg1[%[[IDX]]] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
    %ptr1 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %dst = tt.addptr %ptr1, %off : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %dst, %val : tensor<4x!tt.ptr<f32>>
    
    tt.return
  }
}
