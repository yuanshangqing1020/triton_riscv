
module {
  func.func @test_program_id(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant dense<[0, 1, 2, 3]> : vector<4xi32>
    %0 = vector.broadcast %arg2 : i32 to vector<4xi32>
    %1 = arith.addi %0, %cst : vector<4xi32>
    %c0_i64 = arith.constant 0 : i64
    %2 = vector.broadcast %c0_i64 : i64 to vector<4xi64>
    %3 = arith.extsi %1 : vector<4xi32> to vector<4xi64>
    %4 = arith.addi %2, %3 : vector<4xi64>
    
    // Simulating a load (simplified from previous output)
    %idx = arith.constant 0 : index
    %pad = arith.constant 0.0 : f32
    %vec_load = vector.transfer_read %arg0[%idx], %pad {in_bounds = [true]} : memref<?xf32>, vector<4xf32>
    
    // Simulating a store
    vector.transfer_write %vec_load, %arg1[%idx] {in_bounds = [true]} : vector<4xf32>, memref<?xf32>
    
    return
  }
}
