import ctypes as ct
import torch
import ampere

def get_ptr(A: torch.Tensor) -> ct.c_void_p:
    if A is None:
        return None
    else:
        return ct.c_void_p(A.data.data_ptr())

lib = ct.cdll.LoadLibrary('/home/fernand/sparse_learning/libsparse.so')
lib.get_context.restype = ct.c_void_p
context = ct.c_void_p(lib.get_context())

A = torch.rand((128*128, 4096), dtype=torch.float16, device=torch.device(0))
mask = ampere.create_mask(A)
B = torch.rand((10240, 4096), dtype=torch.float16, device=torch.device(0))
C = torch.zeros((128*128, 10240), dtype=torch.float16, device=torch.device(0))

lib.sparse_matmul(context, get_ptr(A), get_ptr(B), get_ptr(C), A.shape[0], A.shape[1], B.shape[0], B.shape[1])
print((mask * A).matmul(B.T))
print(C)