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
  #ZERO_ERROR -Verified on Jun-23-2016
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
  #FIXME Error not zero 
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
    	raise ValueError("Size mismatch")
    op_list.append(U)	
     
    
  phi = np.random.rand(2**nbits) + np.random.rand(2**nbits)
  phi = phi / np.linalg.norm(phi)
  phi = phi.reshape(2**nbits,1)
  reg = qr.Qreg(nbits,phi.copy())
  reg = Circuit.evaluate(reg)
  
  for V in op_list:
    phi = np.dot(V,phi)
    
  error = np.linalg.norm( phi - reg.array.reshape( 2**nbits, 1))/ float( 2**nbits)
  print reg.array.reshape( 2**nbits, 1)[0]
  return error 
  
  
def check_qcircuit_withcontrol(nbits, circ_length):
  """
    Check the error in Qcircuit by using ranomly generated circuits, without any contrl operations.
    nbits : number of qubits in the circuit
    circ_length :  Length of circuit
    
  """
  #FIXME Unfinished
  Circuit = qc.Qcircuit(nbits)
  op_list = []
  for i in range(circ_length):
    pos = np.random.choice(nbits)
    #print "pos" , pos
    size  = np.random.choice(range(1,nbits - pos+1))
    #print "size", size
    qubitset = range(pos, pos +size)
    if pos != 0:
      control = np.random.choice(range[pos])
    else: 
      control = None  
    A = np.asmatrix( np.random.rand(  2**size, 2**size) + 1j* np.random.rand( 2**size, 2**size))
    A = A + A.H
    U = qo.matpowerh((np.e**1j),A)   
    #print qubitset
    S = U.copy()
    if control == None:
      Circuit.insert_operator(np.asarray(S),qubitset)
      U = np.kron(np.eye(2**pos),U)
      U = np.kron(U, np.eye(2**nbits/U.shape[0]) )
      if U.shape[0] !=  2**nbits:
    	raise ValueError("Size mismatch") 
    else:
      Circuit.insert_operator(np.asarray(S),qubitset) 
      ket1 = qr.getbasis(1,1)
      ket0 = qr.getbasis(1,0)
      mat1 = qr.outerprod(ket1,ket1)
      mat0 = qr.outerprod(ket0, ket0)
      U = np.kron(np.eye(2**pos),U)
      U = np.kron(U, np.eye(2**(control -pos+size -1)))
      U = np.kron(U,ket1)
      U = np.kron(U,np.eye(2**nbits/U.shape[0]))
      E = np.kron(np.eye(2**control),ket0)
      E = np.kron(E,np.eye(2**nbits/E.shape[0]))
      U = U +E
      if U.shape[0] !=  2**nbits:
    	raise ValueError("Size mismatch") 
      
      
      
  
def compare_fourier(nbits,qreg):
   """
    Checking the correctness of Qcircuit operations by implementing the Fourier transform circuit and comparing the result to multiplication by the Fourier operator directly
    """
 	
  


