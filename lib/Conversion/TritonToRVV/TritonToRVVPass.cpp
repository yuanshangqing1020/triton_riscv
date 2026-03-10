#include "triton/Conversion/TritonToRVV/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONTORVV
#include "triton/Conversion/TritonToRVV/Passes.h.inc"
}
}

using namespace mlir;
using namespace mlir::triton;

namespace {

class TritonToRVVTypeConverter : public TypeConverter {
public:
  TritonToRVVTypeConverter() {
    addConversion([](Type type) { return type; });
    
    // Convert Triton pointers to MemRefs
    // We treat pointers as dynamic 1D memrefs: memref<?xEltTy>
    addConversion([&](triton::PointerType ptrType) -> std::optional<Type> {
      return MemRefType::get({ShapedType::kDynamic}, ptrType.getPointeeType());
    });

    // Convert Tensors to Vectors
    addConversion([&](RankedTensorType tensorType) -> std::optional<Type> {
      Type eltType = tensorType.getElementType();
      // For now, if we encounter a tensor of pointers, we treat it as vector of i64 (addresses)
      // This is a placeholder until we have a robust strategy for block pointers
      if (llvm::isa<triton::PointerType>(eltType)) {
         return VectorType::get(tensorType.getShape(), IntegerType::get(tensorType.getContext(), 64)); 
      }
      return VectorType::get(tensorType.getShape(), eltType);
    });
  }
};

struct FuncOpConversion : public OpConversionPattern<triton::FuncOp> {
  using OpConversionPattern<triton::FuncOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto funcType = op.getFunctionType();
    TypeConverter::SignatureConversion signatureConversion(funcType.getNumInputs());
    if (failed(typeConverter->convertSignatureArgs(funcType.getInputs(), signatureConversion)))
      return failure();

    // Convert result types
    SmallVector<Type> newResultTypes;
    if (failed(typeConverter->convertTypes(funcType.getResults(), newResultTypes)))
      return failure();

    auto newFuncType = FunctionType::get(rewriter.getContext(), signatureConversion.getConvertedTypes(),
                                         newResultTypes);

    auto newFuncOp = func::FuncOp::create(op.getLoc(), op.getName(), newFuncType);
    rewriter.insert(newFuncOp);
    rewriter.inlineRegionBefore(op.getBody(), newFuncOp.getBody(), newFuncOp.end());
    
    if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter, &signatureConversion)))
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

struct ReturnOpConversion : public OpConversionPattern<triton::ReturnOp> {
  using OpConversionPattern<triton::ReturnOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

struct SplatOpConversion : public OpConversionPattern<triton::SplatOp> {
  using OpConversionPattern<triton::SplatOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::SplatOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType) return failure();
    rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, resultType, adaptor.getSrc());
    return success();
  }
};

struct ArithConstantOpConversion : public OpConversionPattern<arith::ConstantOp> {
  using OpConversionPattern<arith::ConstantOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto resultType = typeConverter->convertType(op.getType());
    if (!resultType || !isa<VectorType>(resultType)) return failure();

    auto value = op.getValue();
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
      auto newType = cast<ShapedType>(resultType);
      rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, resultType, denseAttr.reshape(newType));
      return success();
    }
    return failure();
  }
};

template <typename SourceOp, typename DestOp>
struct ArithOpConversion : public OpConversionPattern<SourceOp> {
  using OpConversionPattern<SourceOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<DestOp>(op, adaptor.getOperands());
    return success();
  }
};

class ConvertTritonToRVV : public triton::impl::ConvertTritonToRVVBase<ConvertTritonToRVV> {
public:
  void runOnOperation() override {
    ModuleOp module = getOperation();
    TritonToRVVTypeConverter typeConverter;
    
    ConversionTarget target(getContext());
    target.addLegalDialect<scf::SCFDialect,
                           vector::VectorDialect, memref::MemRefDialect,
                           func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    
    // Mark arith ops on tensors as illegal
    target.addDynamicallyLegalOp<arith::AddFOp, arith::SubFOp, arith::MulFOp,
                                 arith::AddIOp, arith::SubIOp, arith::MulIOp,
                                 arith::ConstantOp>(
        [](Operation *op) {
          return !llvm::any_of(op->getResultTypes(), [](Type type) { return llvm::isa<TensorType>(type); });
        });

    target.addIllegalDialect<triton::TritonDialect>();
    target.addLegalOp<ModuleOp>(); 

    RewritePatternSet patterns(&getContext());
    // Pass &getContext() because OpConversionPattern expects MLIRContext*
    patterns.add<FuncOpConversion>(typeConverter, &getContext());
    patterns.add<ReturnOpConversion>(typeConverter, &getContext());
    patterns.add<SplatOpConversion>(typeConverter, &getContext());
    patterns.add<ArithConstantOpConversion>(typeConverter, &getContext());
    
    // Arithmetic ops
    patterns.add<ArithOpConversion<arith::AddFOp, arith::AddFOp>>(typeConverter, &getContext());
    patterns.add<ArithOpConversion<arith::SubFOp, arith::SubFOp>>(typeConverter, &getContext());
    patterns.add<ArithOpConversion<arith::MulFOp, arith::MulFOp>>(typeConverter, &getContext());
    patterns.add<ArithOpConversion<arith::AddIOp, arith::AddIOp>>(typeConverter, &getContext());
    patterns.add<ArithOpConversion<arith::SubIOp, arith::SubIOp>>(typeConverter, &getContext());
    patterns.add<ArithOpConversion<arith::MulIOp, arith::MulIOp>>(typeConverter, &getContext());

    // TODO: Add load/store patterns

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
