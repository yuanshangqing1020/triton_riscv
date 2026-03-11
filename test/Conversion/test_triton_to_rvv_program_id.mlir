// RUN: triton-opt %s -convert-triton-to-rvv | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_program_id
  // CHECK-SAME: (%[[ARG0:.*]]: memref<?xf32>, %[[ARG1:.*]]: memref<?xf32>, %[[PID_X:.*]]: i32, %[[PID_Y:.*]]: i32, %[[PID_Z:.*]]: i32)
  tt.func @test_program_id(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
    // CHECK: %[[CST:.*]] = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    
    // CHECK: %[[PID_X_VEC:.*]] = vector.broadcast %[[PID_X]] : i32 to vector<4xi32>
    %pid = tt.get_program_id x : i32
    %pid_vec = tt.splat %pid : i32 -> tensor<4xi32>
    
    // CHECK: %[[ADD:.*]] = arith.addi %[[PID_X_VEC]], %[[CST]] : vector<4xi32>
    %offset = arith.addi %pid_vec, %0 : tensor<4xi32>
    
    // CHECK: %[[C0:.*]] = arith.constant 0 : i64
    // CHECK: %[[SPLAT:.*]] = vector.broadcast %[[C0]] : i64 to vector<4xi64>
    // CHECK: %[[EXT:.*]] = arith.extsi %[[ADD]] : vector<4xi32> to vector<4xi64>
    // CHECK: %[[PTR_ADD:.*]] = arith.addi %[[SPLAT]], %[[EXT]] : vector<4xi64>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.addptr %1, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    
    %3 = tt.load %2 : tensor<4x!tt.ptr<f32>>
    
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %5 = tt.addptr %4, %offset : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %5, %3 : tensor<4x!tt.ptr<f32>>

    tt.return
  }
}
