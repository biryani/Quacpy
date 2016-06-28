import qregister as qr
import qoperator as qo
import qcircuit as qc
import numpy as np

"""
Test functions for Quacpy
"""
def  sub_slice(array, dims, indices):
  """
   Selects a subarray from a multidimensionalarray, with the values given by 
   indicies in corresponding positions given by dims. indicies cannot be ranges, only values.
   Required for implementing control operations.
  """


  ndim = array.ndim
  ind_list = []
  for i in range(ndim):
    if i in dims:
      j  = dims.index(i)
      ind_list.append(slice(indices[j], indices[j]+1, None))
    else:
      ind_list.append(slice(None,))       
     
  return array[ind_list]

def compare_einsum_dot(n,k,d = 0):
  """
   Proof of concept tests to check if the einsum based tensor contraction gives    the same result as multiplication with an operaqtor of the form I \otimes U.
   returns the error between the two implementations  
   
  """
  #ZERO_ERROR - for all d >= 0 Verified on Jun-24-2016
 
  phi = np.random.random_sample([2]*n) + 1j*np.random.random_sample([2]*n)
  U =  np.random.random_sample([2]*(2*k)) + 1j*np.random.random_sample([2]*(2*k))
  M = np.kron(np.eye(2**(n-k-d)),U.reshape(2**k,2**k))
  M = np.kron(M, np.eye(2**d))
  #print M.shape, 2**n
  x = phi.flatten().copy()
  t1 = np.dot(M, x)
  ind = range(n)
  indU = ind[-k-d:-d]
  if d == 0: indU = ind[-k:]
  #print ind
  indU =  range(n,n+k) + indU 
  indout = range(n) 
  if d == 0:
    indout[-k:] = range(n,n+k)	
  else:
    indout[-k-d:-d] = range(n, n+k) #outputshape
 
  #print indU
  t2 = np.einsum( U, indU, phi, ind, indout)
  #print t2.shape
  return np.linalg.norm(t1 - t2.flatten())
 
def check_qcircuit_nocntrl(nbits, circ_length):
  """
    Check the error in Qcircuit by using ranomly generated circuits, without any contrl operations.
    nbits : number of qubits in the circuit
    circ_length :  Length of circuit
    
  """
  #ZERO_ERROR - Verified on Jun-24-2016
  Circuit = qc.Qcircuit(nbits)
  op_list = []
  for i in range(circ_length):
    pos = np.random.choice(nbits)
    print "pos" , pos
    size  = np.random.choice(range(1,nbits - pos+1))
    print "size", size
    qubitset = range(pos, pos +size)
    A = np.asmatrix( np.random.rand(  2**size, 2**size) + 1j* np.random.rand( 2**size, 2**size))
    A = A + A.H
    U = qo.matpowerh((np.e**1j),A)   
    #print qubitset
    S = U.copy()
    Circuit.insert_operator(np.asarray(S),qubitset)
    
    U = np.kron(np.eye(2**pos),U)
    print 2**nbits/U.shape[0]
    U = np.kron(U, np.eye(2**nbits/U.shape[0]) )
   
    if U.shape[0] !=  2**nbits:
    	raise ValueError("Size mismatch")
    op_list.append(U)	
     
    
  phi = np.random.rand(2**nbits) + 1j*np.random.rand(2**nbits)
  phi = phi / np.linalg.norm(phi)
  phi = phi.reshape(2**nbits,1)
  reg = qr.Qreg(nbits,phi)
  
  reg = Circuit.evaluate(reg)
  
  for V in op_list:
    phi = np.dot(V,phi)
    
  error = np.linalg.norm( phi - reg.array.reshape( 2**nbits, 1))/ float( 2**nbits)
  
  return error#, phi, reg.array, U
  
  
def check_qcircuit_withcontrol(nbits, circ_length):
  """
    Check the error in Qcircuit by using ranomly generated circuits, without any contrl operations.
    nbits : number of qubits in the circuit
    circ_length :  Length of circuit
    
  """
  #ZERO_ERROR - Verified on Jun-24-2016
  Circuit = qc.Qcircuit(nbits)
  op_list = []
  for i in range(circ_length):
    pos = np.random.choice(nbits)
    print "pos" , pos
    size  = np.random.choice(range(1,nbits - pos+1))
    print "size", size
    qubitset = range(pos, pos +size)
    if pos != 0:
      control = np.random.choice(range(pos))
    else: 
      control = None
    print "control", control    
    A = np.asmatrix( np.random.rand(  2**size, 2**size) + 1j* np.random.rand( 2**size, 2**size))
    A = A + A.H
    U = qo.matpowerh((np.e**1j),A)   
    print U.shape
    #print qubitset
    S = U.copy()
    if control == None:
      Circuit.insert_operator(np.asarray(S),qubitset)
      U = np.kron(np.eye(2**pos),U)
      U = np.kron(U, np.eye(2**nbits/U.shape[0]) )
      if U.shape[0] !=  2**nbits:
    	raise ValueError("Size mismatch") 
    else:
      Circuit.insert_operator(np.asarray(S),qubitset, cntrl1 = [control]) 
      ket1 = qr.getbasis(1,1)
     
      ket0 = qr.getbasis(1,0)
      mat1 = qr.outerprod(ket1,ket1)
      mat0 = qr.outerprod(ket0, ket0)
      V = np.kron(np.eye(2**control),mat1)
      V = np.kron(V, np.eye(2**(pos-control-1)))
      print U.shape, V.shape
      U = np.kron(V,U)
      U = E = np.kron(U,np.eye(2**nbits/U.shape[0]))
    
      E = np.kron(np.eye(2**control),mat0)
      E = np.kron(E,np.eye(2**nbits/E.shape[0]))
      print E.shape
      U = U +E
      if U.shape[0] !=  2**nbits:
    	raise ValueError("Size mismatch") 
      
    op_list.append(U)
     
  phi = np.random.rand(2**nbits) + 1j*np.random.rand(2**nbits)
  phi = phi / np.linalg.norm(phi)
  phi = phi.reshape(2**nbits,1)
  reg = qr.Qreg(nbits,phi.copy())
  
  reg = Circuit*reg ##operator oveloading!!
  
  for V in op_list:
    phi = np.dot(V,phi)
    
  error = np.linalg.norm( phi - reg.array.reshape( 2**nbits, 1))/ float( 2**nbits)
  
  return error
     
def H_op():
  return (1/np.sqrt( 2))*np.array([[1, 1], [1, -1]], dtype = complex)
    
def R_op(theta):
  return np.array([[1, 0], [0,np.exp(1j*theta)]], dtype = complex)      
      
  
def fourier(nbits):
  """
    Checking the correctness of Qcircuit operations by implementing the Fourier transform circuit and comparing the result to multiplication by the Fourier operator directly
  """
  n  =  nbits
##TODO IMplement from Mike and Ike 
  if n ==1 :
    F = qc.Qcircuit(1)
    F.insert_operator(H_op(), [0])
    return F
   
    
  elif n > 1 :
    F = fourier( nbits -1)
    F.nbits += 1
    for i in range(n-1,0,-1):
      F.insert_operator(R_op(2*np.pi/float(2**(i+1))), [n-1], cntrl1 = [0])
      
    F.insert_operator(H_op(), [n-1])
    return F        
  
  else:
    raise ValueError("number of qubits should be an integer greater than or equal to 1")
    return
   
   
   

def reverser(nbits):
  #24_jun-16: Seems to work 
  M = np.zeros((4,4), dtype = np.complex128)
  M[0,0] = 1.0
  M[3,3] = 1.0
  M[1,2] = 1.0
  M[2,1] = 1.0 
  if nbits == 2:
    S = qc.Qcircuit(2)
  
    S.insert_operator(M, [0,1])
    return S
    
  if nbits > 2:
    S = reverser(nbits-1)
    S.nbits += 1
    for i in range(nbits-1,0,-1):
      S.insert_operator(M, [i, i-1])
      
    return S
    
      
        



         
    
         	
  


