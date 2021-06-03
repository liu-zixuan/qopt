import numpy as np
import itertools

def _tensor_perm(dims, perm):
    p = np.empty(dims[perm], dtype=perm.dtype)
    for i, t in enumerate(itertools.product(*[np.arange(d) for d in dims])):
        p[tuple(np.array(t)[perm])] = i
    return p.flatten()


'''
permute the tensor order of a composite system
A: the operator, numpy.array
dims: dimensions of the sub-systems, list or numpy.array
perm: the permutation vector
inv (default False): invert the permutation vector
'''
def permute_systems(A, dims, perm, inv=False):
    dims = np.array(dims)
    perm = np.array(perm)
    if inv is True:
        perm_inv = np.empty_like(perm)
        perm_inv[perm] = np.arange(len(perm_inv), dtype=perm_inv.dtype)
        perm = perm_inv
    p = _tensor_perm(dims, perm)
    return (A[:, p])[p]


'''
partial trace
A: the operator, numpy.array
dims: dimensions of the sub-systems, list or numpy.array
trace_list: the systems to be traced out, list or numpy.array
'''
def partial_trace(A, dims, trace_list):
    trace_list.sort()
    traced = 0
    for t in itertools.product(*[np.arange(dims[i]) for i in trace_list]):
        # construct the kraus operators
        s = 0
        K = 1
        for i, d in enumerate(dims):
            if i in trace_list:
                P = np.zeros((1,d))
                P[0, t[s]] = 1
                s += 1
                K = np.kron(K, P)
            else:
                K = np.kron(K, np.eye(d))
        traced += K @ A @ K.T
    return traced


'''
convert a unitary matrix to the choi operator
U: the operator, numpy.array
'''
def choi(U):
    vec_U = U.T.reshape(U.size, 1)
    return  vec_U @ vec_U.T.conj()