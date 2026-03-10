import torch
import triton
import triton.language as tl
from triton.compiler import compile, ASTSource

@triton.jit
def kernel(ptr):
    a = tl.full((128,), 1.0, dtype=tl.float32)
    b = tl.full((128,), 2.0, dtype=tl.float32)
    # This will be folded to 3.0 by the frontend
    c = a + b 
    
    # This will generate pointers and stores which are not yet supported in Phase 3
    offs = tl.arange(0, 128)
    tl.store(ptr + offs, c)

# Create ASTSource object
src = ASTSource(fn=kernel, signature={"ptr": "*fp32"}, constexprs={})
# Compile to TTIR
ret = compile(src, target=None, options={"only_ttir": True})
print(ret.asm["ttir"])
