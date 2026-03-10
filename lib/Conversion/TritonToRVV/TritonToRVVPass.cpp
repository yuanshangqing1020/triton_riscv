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

// Helper to avoid deprecated OpBuilder::create
template <typename OpTy, typename... Args>
OpTy createOp(OpBuilder &builder, Location loc, Args &&...args) {
    auto op = OpTy::create(builder, loc, std::forward<Args>(args)...);
    // OpTy::create already inserts the operation using the builder.
    return op;
}

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
    
    if (isa<MemRefType>(adaptor.getSrc().getType())) {
        auto vecType = cast<VectorType>(resultType);
        auto zeroAttr = rewriter.getIntegerAttr(vecType.getElementType(), 0);
        auto zero = createOp<arith::ConstantOp>(rewriter, op.getLoc(), zeroAttr);
        rewriter.replaceOpWithNewOp<vector::BroadcastOp>(op, resultType, zero);
        return success();
    }
 
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

struct MakeRangeOpConversion : public OpConversionPattern<triton::MakeRangeOp> {
  using OpConversionPattern<triton::MakeRangeOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::MakeRangeOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    int32_t start = op.getStart();
    int32_t end = op.getEnd();
    auto resultType = typeConverter->convertType(op.getType());
    auto vectorType = dyn_cast<VectorType>(resultType);
    if (!vectorType) return failure();

    SmallVector<Attribute> values;
    for (int32_t i = start; i < end; ++i) {
      values.push_back(rewriter.getI32IntegerAttr(i));
    }
    
    auto attr = DenseElementsAttr::get(vectorType, values);
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, vectorType, attr);
    return success();
  }
};

struct AddPtrOpConversion : public OpConversionPattern<triton::AddPtrOp> {
  using OpConversionPattern<triton::AddPtrOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::AddPtrOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    if (isa<MemRefType>(adaptor.getPtr().getType())) {
        return failure();
    }
    Value ptr = adaptor.getPtr();
    Value offset = adaptor.getOffset();
    
    auto ptrType = cast<VectorType>(ptr.getType());
    auto offsetType = cast<VectorType>(offset.getType());
    
    if (ptrType.getElementType() != offsetType.getElementType()) {
        // Sign extend offset to match pointer width
        if (ptrType.getElementType().getIntOrFloatBitWidth() > offsetType.getElementType().getIntOrFloatBitWidth()) {
             offset = createOp<arith::ExtSIOp>(rewriter, op.getLoc(), ptrType, offset);
        }
    }

    rewriter.replaceOpWithNewOp<arith::AddIOp>(op, ptr, offset);
    return success();
  }
};

struct LoadOpConversion : public OpConversionPattern<triton::LoadOp> {
  using OpConversionPattern<triton::LoadOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::LoadOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value base;
    Value offset;
    
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    if (addPtrOp) {
        auto splatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
        if (splatOp) {
            base = rewriter.getRemappedValue(splatOp.getSrc());
            offset = rewriter.getRemappedValue(addPtrOp.getOffset());
        }
    }
    
    if (!base || !offset || !isa<MemRefType>(base.getType())) {
        return failure();
    }
    
    auto loc = op.getLoc();
    auto resultType = typeConverter->convertType(op.getType());
    auto vecType = cast<VectorType>(resultType);
    auto elemType = vecType.getElementType();
    
    SmallVector<int64_t> pos = {0};
    Value startOffset = createOp<vector::ExtractOp>(rewriter, loc, offset, pos);
    
    if (!isa<IndexType>(startOffset.getType())) {
        startOffset = createOp<arith::IndexCastOp>(rewriter, loc, rewriter.getIndexType(), startOffset);
    }
    
    Value padding;
    if (isa<FloatType>(elemType)) {
        padding = createOp<arith::ConstantOp>(rewriter, loc, rewriter.getFloatAttr(elemType, 0.0));
    } else {
        padding = createOp<arith::ConstantOp>(rewriter, loc, rewriter.getIntegerAttr(elemType, 0));
    }
    
    SmallVector<Value> indices = {startOffset};
    SmallVector<bool> inBounds(vecType.getRank(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBounds);
    auto map = AffineMap::getMultiDimIdentityMap(vecType.getRank(), rewriter.getContext());
    
    rewriter.replaceOpWithNewOp<vector::TransferReadOp>(
        op, resultType, base, indices, AffineMapAttr::get(map), padding, /*mask=*/Value(), inBoundsAttr);
        
    return success();
  }
};

struct StoreOpConversion : public OpConversionPattern<triton::StoreOp> {
  using OpConversionPattern<triton::StoreOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(triton::StoreOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Value ptr = op.getPtr();
    Value base;
    Value offset;
    
    auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>();
    if (addPtrOp) {
        auto splatOp = addPtrOp.getPtr().getDefiningOp<triton::SplatOp>();
        if (splatOp) {
            base = rewriter.getRemappedValue(splatOp.getSrc());
            offset = rewriter.getRemappedValue(addPtrOp.getOffset());
        }
    }
    
    if (!base || !offset || !isa<MemRefType>(base.getType())) {
        return failure();
    }
    
    auto loc = op.getLoc();
    SmallVector<int64_t> pos = {0};
    Value startOffset = createOp<vector::ExtractOp>(rewriter, loc, offset, pos);
    
    if (!isa<IndexType>(startOffset.getType())) {
        startOffset = createOp<arith::IndexCastOp>(rewriter, loc, rewriter.getIndexType(), startOffset);
    }
    
    SmallVector<Value> indices = {startOffset};
    auto vecType = cast<VectorType>(adaptor.getValue().getType());
    SmallVector<bool> inBounds(vecType.getRank(), true);
    auto inBoundsAttr = rewriter.getBoolArrayAttr(inBounds);
    auto map = AffineMap::getMultiDimIdentityMap(vecType.getRank(), rewriter.getContext());
    
    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, adaptor.getValue(), base, indices, AffineMapAttr::get(map), /*mask=*/Value(), inBoundsAttr);
        
    return success();
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
    patterns.add<MakeRangeOpConversion>(typeConverter, &getContext());
    patterns.add<AddPtrOpConversion>(typeConverter, &getContext());
    patterns.add<LoadOpConversion>(typeConverter, &getContext());
    patterns.add<StoreOpConversion>(typeConverter, &getContext());
    
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
