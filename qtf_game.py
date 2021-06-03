"""
Show the advantages of the quantum time flip over non-causal network in a quantum game
"""

import numpy as np
import cvxpy as cp
from qtoolbox import *

# partial trace and replace the traced systems with maximally mixed states
def partial_erasure(A, dims, trace_list):
    perm = trace_list + [i for i in range(len(dims)) if i not in trace_list]
    new_dims = [dims[i] for i in perm]
    td = np.prod([dims[i] for i in trace_list])
    return permute_systems(cp.kron(np.eye(td)/td, partial_trace(A, dims, trace_list)), new_dims, perm, inv=True)

d = 2
dims = [d] * 4
# Pauli gates
I = np.array([[1,0], [0,1]])
X = np.array([[0,1], [1,0]])
Y = np.array([[0,-1j], [1j,0]])
Z = np.array([[1,0], [0,-1]])


S0 = [
    (I, I), (I, X), (I, Z),
    (X, I), (X, X), (X, Z),
    (Z, I), (Z, X), (Z, Z),
    (X+Y, X-Y), (X-Y, X+Y),
    (Z+Y, Z-Y), (Z-Y, Z+Y),
]
S0 = [np.kron(choi(p[0]), choi(p[1])) for p in S0]
S0 = [x * d**2 / np.trace(x) for x in S0]

S1 = [
    (Y, I), (Y, X), (Y, Z),
    (I, Y), (X, Y), (Z, Y),
    (I+1j*Y, I-1j*Y), (I-1j*Y, I+1j*Y),
]
S1 = [np.kron(choi(p[0]), choi(p[1])) for p in S1]
S1 = [x * d**2 / np.trace(x) for x in S1]

# C is a non-causal network
C = cp.Variable((d**4, d**4), hermitian=True)
# {P, C-P} is a non-causal tester
P = cp.Variable((d**4, d**4), hermitian=True)
p_err = cp.Variable(len(S0) + len(S1))

"""
Slot 1: 0 1
Slot 2: 2 3
the constraints for a non-causal network
C + t13_C == t1_C + t3_C
t01_C == t013_C
t23_C == t123_C
"""
t1_C = partial_erasure(C, dims, [1])
t3_C = partial_erasure(C, dims, [3])
t13_C = partial_erasure(C, dims, [1,3])
t01_C = partial_erasure(C, dims, [0,1])
t23_C = partial_erasure(C, dims, [2,3])
t013_C = partial_erasure(C, dims, [0,1,3])
t123_C = partial_erasure(C, dims, [1,2,3])

cons = [
    C + t13_C == t1_C + t3_C,
    t01_C == t013_C,
    t23_C == t123_C,
    cp.trace(C) == d**2,
    C >> P,
    P >> 0,
]
cons += [p_err[i] == cp.real(1 - cp.trace(P @ np.array(S0[i]))) for i in range(len(S0))]
cons += [p_err[len(S0) + i] == cp.real(cp.trace(P @ np.array(S1[i]))) for i in range(len(S1))]

prob = cp.Problem(cp.Minimize(cp.max(p_err)), cons)
prob.solve(solver=cp.MOSEK)

print("The optimal value is", prob.value)
print("Status:", prob.status)