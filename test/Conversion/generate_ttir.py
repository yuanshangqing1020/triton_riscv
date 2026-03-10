import torch
import triton
import triton.language as tl
from triton.compiler import compile, ASTSource

@triton.jit
def kernel(ptr):
    a = tl.full((128,), 1.0, dtype=tl.float32)
    b = tl.full((128,), 2.0, dtype=tl.float32)
    c = a + b
    # Ensure usage to prevent dead code elimination
    offs = tl.arange(0, 128)
    tl.store(ptr + offs, c)

src = ASTSource(fn=kernel, signature={"ptr": "*fp32"}, constexprs={})
ret = compile(src, target=None, options={"only_ttir": True})
print(ret.asm["ttir"])
