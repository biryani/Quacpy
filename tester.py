import qregister as qr
import qoperator as qo
import qcircuit as qc
import numpy as np

"""
Test functions for Quacpy
"""
def compare_einsum_dot(n,k):
  """
   Proof of concept tests to check if the einsum based tensor contraction gives    the same result as multiplication with an operaqtor of the form I \otimes U.
   returns the error between the two implementations  
  """

  phi = np.random.random_sample([2]*n)
  U =  np.random.random_sample([2]*(2*k))
  M = np.kron(np.eye(2**(n-k)),U.reshape(2**k,2**k))
  x = phi.flatten()
  t1 = np.dot(M, x)
  ind = range(n)
  indU = ind[-k:]
  indU =  range(n,n+k) + indU
  t2 = np.einsum( U, indU, phi, ind)
  return np.linalg.norm(t1 - t2.flatten())


