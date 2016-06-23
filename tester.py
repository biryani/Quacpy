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
 
def check_qcircuit_nocntrl(nbits, circ_length):
  """
    Check the error in Qcircuit by using ranomly generated circuits, without any contrl operations.
    nbits : number of qubits in the circuit
    circ_length :  Length of circuit
    
  """
  Circuit = qc.Qcircuit(nbits)
  op_list = []
  for i in range(circ_length):
    pos = np.random.choice(nbits)
    #print "pos" , pos
    size  = np.random.choice(range(1,nbits - pos+1))
    #print "size", size
    qubitset = range(pos, pos +size)
    A = np.asmatrix( np.random.rand(  2**size, 2**size) + 1j* np.random.rand( 2**size, 2**size))
    A = A + A.H
    U = qo.matpowerh((np.e**1j),A)   
    #print qubitset
    S = U.copy()
    Circuit.insert_operator(np.asarray(S),qubitset)
    
    U = np.kron(np.eye(2**pos),U)
    U = np.kron(U, np.eye(2**nbits/U.shape[0]) )
    if U.shape[0] !=  2**nbits:
    	raise ValueError("Not good!")
     
    
  phi = np.random.rand(2**nbits) + np.random.rand(2**nbits)
  phi = phi / np.linalg.norm(phi)
  phi = phi.reshape(2**nbits,1)
  reg = qr.Qreg(nbits,phi)
  reg = Circuit.evaluate(reg)
  
  for V in op_list:
    phi = np.dot(V,phi)
    
  error = np.linalg.norm( phi - reg.array.reshape( 2**nbits, 1))/ float( 2**nbits)
  return error     
  
def compare_fourier(nbits,qreg):
   """
    Checking the correctness of Qcircuit operations by implementing the Fourier transform circuit and compating the result to multiplication by the Fourier operator directly
    """
 	
  


