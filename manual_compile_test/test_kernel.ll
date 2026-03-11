; ModuleID = 'LLVMDialectModule'
source_filename = "LLVMDialectModule"

define void @test_program_id(ptr %0, ptr %1, i64 %2, i64 %3, i64 %4, ptr %5, ptr %6, i64 %7, i64 %8, i64 %9, i32 %10, i32 %11, i32 %12) {
  %14 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %5, 0
  %15 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %14, ptr %6, 1
  %16 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %15, i64 %7, 2
  %17 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %16, i64 %8, 3, 0
  %18 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %17, i64 %9, 4, 0
  %19 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } poison, ptr %0, 0
  %20 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %19, ptr %1, 1
  %21 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %20, i64 %2, 2
  %22 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %21, i64 %3, 3, 0
  %23 = insertvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %22, i64 %4, 4, 0
  %24 = insertelement <4 x i32> poison, i32 %10, i32 0
  %25 = shufflevector <4 x i32> %24, <4 x i32> poison, <4 x i32> zeroinitializer
  %26 = add <4 x i32> %25, <i32 0, i32 1, i32 2, i32 3>
  %27 = extractelement <4 x i32> %26, i64 0
  %28 = sext i32 %27 to i64
  %29 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %23, 1
  %30 = getelementptr float, ptr %29, i64 %28
  %31 = load <4 x float>, ptr %30, align 4
  %32 = extractelement <4 x i32> %26, i64 0
  %33 = sext i32 %32 to i64
  %34 = extractvalue { ptr, ptr, i64, [1 x i64], [1 x i64] } %18, 1
  %35 = getelementptr float, ptr %34, i64 %33
  store <4 x float> %31, ptr %35, align 4
  ret void
}

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
