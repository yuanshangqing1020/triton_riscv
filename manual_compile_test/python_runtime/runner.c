
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <string.h>

// Define the signature of the kernel function
// void test_program_id(ptr, ptr, i64, i64, i64, ptr, ptr, i64, i64, i64, i32, i32, i32)
typedef void (*kernel_func_t)(
    void*, void*, int64_t, int64_t, int64_t, 
    void*, void*, int64_t, int64_t, int64_t, 
    int32_t, int32_t, int32_t
);

int main(int argc, char** argv) {
    // For static linking test:
    // Declare the function as external
    extern void test_program_id(
        void*, void*, int64_t, int64_t, int64_t, 
        void*, void*, int64_t, int64_t, int64_t, 
        int32_t, int32_t, int32_t
    );
    kernel_func_t kernel = test_program_id;

/*
    // Dynamic loading (commented out for Spike PK static linking)
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_so>\n", argv[0]);
        return 1;
    }

    const char* lib_path = argv[1];
    void* handle = dlopen(lib_path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "Error loading library: %s\n", dlerror());
        return 1;
    }

    kernel_func_t kernel = (kernel_func_t)dlsym(handle, "test_program_id");
    if (!kernel) {
        fprintf(stderr, "Error finding symbol 'test_program_id': %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
*/

    // Allocate memory for inputs and outputs
    // We need 2 pointers for input and output buffers (arg0, arg1)
    // And other pointers (arg5, arg6)
    
    // Based on the kernel logic:
    // It reads from arg1[idx] and writes to arg5[idx]
    // where idx comes from program_id + vector math
    
    // Let's allocate some dummy buffers
    float* in_ptr = (float*)aligned_alloc(64, 1024 * sizeof(float));
    float* out_ptr = (float*)aligned_alloc(64, 1024 * sizeof(float));
    
    // Initialize input
    for (int i = 0; i < 1024; i++) {
        in_ptr[i] = (float)i;
        out_ptr[i] = -1.0f; // Sentinel
    }

    // Prepare arguments
    // The kernel signature in MLIR:
    // %arg0: ptr, %arg1: ptr, %arg2: i64, %arg3: i64, %arg4: i64
    // %arg5: ptr, %arg6: ptr, %arg7: i64, %arg8: i64, %arg9: i64
    // %arg10: i32, %arg11: i32, %arg12: i32
    
    // The kernel uses:
    // arg11 (struct ptr) -> extract value -> base pointer for LOAD
    // arg5 (struct ptr) -> extract value -> base pointer for STORE
    // arg10 (program_id)
    
    // IMPORTANT: The MLIR uses llvm.insertvalue to build structs from raw pointers.
    // The C-ABI for passing these structs (or expanded arguments) depends on how `llc` lowered it.
    // However, looking at the MLIR signature:
    // llvm.func @test_program_id(%arg0: !llvm.ptr, ...
    // These are scalar arguments.
    
    // Let's assume standard C calling convention for scalars.
    
    // We need to map which argument corresponds to which buffer.
    // In MLIR:
    // %21 = extractvalue %11[1]  -> %11 comes from arg4. No wait.
    // Let's trace back carefully.
    
    // %11 = insertvalue %arg4, %10[4, 0] ... wait, this is building a struct.
    // %11 is the result of a chain of insertvalues starting from %arg0...%arg4
    // Actually, looking at the MLIR:
    // %7 = insertvalue %arg0 ...
    // %8 = insertvalue %arg1 ...
    // ...
    // %11 = insertvalue %arg4 ...
    // So %11 represents the struct built from arg0, arg1, arg2, arg3, arg4.
    // %21 = extractvalue %11[1] -> This gets the element at index 1 of the struct.
    // The struct type is !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
    // Index 1 is the second element, which is a ptr.
    // This corresponds to %arg1.
    
    // So Input Pointer is %arg1.
    
    // Similarly for Output:
    // %5 = insertvalue %arg9 ...
    // %5 represents struct built from arg5...arg9.
    // %27 = extractvalue %5[1] -> Element at index 1.
    // This corresponds to %arg6.
    
    // So Output Pointer is %arg6.
    
    // Program ID is %arg10.
    
    // Invoking Kernel
    printf("Invoking kernel...\n");
    kernel(
        NULL, in_ptr, 0, 0, 0,       // arg0-arg4 (Input Tensor Desc) -> arg1 is the pointer
        NULL, out_ptr, 0, 0, 0,      // arg5-arg9 (Output Tensor Desc) -> arg6 is the pointer
        0, 0, 0                      // arg10-arg12 (Grid X, Y, Z) -> arg10 is X
    );

    // Verify output
    // The kernel does:
    // %17 = program_id(0) + [0, 1, 2, 3]
    // Load from in_ptr[%17]
    // Store to out_ptr[%17]
    
    // Since program_id(0) passed is 0.
    // It should copy elements 0, 1, 2, 3.
    
    printf("Verifying output...\n");
    int errors = 0;
    for (int i = 0; i < 4; i++) {
        float expected = (float)i;
        float actual = out_ptr[i];
        if (actual != expected) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, actual);
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("Verification SUCCESS!\n");
    } else {
        printf("Verification FAILED with %d errors.\n", errors);
    }

/*
    free(in_ptr);
    free(out_ptr);
    dlclose(handle);
    return errors > 0 ? 1 : 0;
*/
    free(in_ptr);
    free(out_ptr);
    return errors > 0 ? 1 : 0;
}
