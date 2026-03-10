// RUN: triton-opt %s -convert-triton-to-rvv | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_load_store
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>)
  tt.func @test_load_store(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    // CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    
    // CHECK: %[[C0:.*]] = arith.constant 0 : i64
    // CHECK: %[[SPLAT:.*]] = vector.broadcast %[[C0]] : i64 to vector<4xi64>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[CST]] : vector<4xi32> to vector<4xi64>
    // CHECK: %[[ADD:.*]] = arith.addi %[[SPLAT]], %[[EXT]] : vector<4xi64>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.addptr %1, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    
    // CHECK: %[[EXTRACT:.*]] = vector.extract %[[CST]][0] : i32 from vector<4xi32>
    // CHECK: %[[IDX:.*]] = arith.index_cast %[[EXTRACT]] : i32 to index
    // CHECK: %[[PAD:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[LOAD:.*]] = vector.transfer_read %[[ARG0]][%[[IDX]]], %[[PAD]] {in_bounds = [true]} : memref<?xf32>, vector<4xf32>
    %3 = tt.load %2 : tensor<4x!tt.ptr<f32>>
    
    // CHECK: %[[C0_2:.*]] = arith.constant 0 : i64
    // CHECK: %[[SPLAT_2:.*]] = vector.broadcast %[[C0_2]] : i64 to vector<4xi64>
    // CHECK: %[[EXT_2:.*]] = arith.extsi %[[CST]] : vector<4xi32> to vector<4xi64>
    // CHECK: %[[ADD_2:.*]] = arith.addi %[[SPLAT_2]], %[[EXT_2]] : vector<4xi64>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    
    // CHECK: %[[EXTRACT_2:.*]] = vector.extract %[[CST]][0] : i32 from vector<4xi32>
    // CHECK: %[[IDX_2:.*]] = arith.index_cast %[[EXTRACT_2]] : i32 to index
    // CHECK: vector.transfer_write %[[LOAD]], %[[ARG1]][%[[IDX_2]]] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
    tt.store %5, %3 : tensor<4x!tt.ptr<f32>>
    
    tt.return
  }
}
