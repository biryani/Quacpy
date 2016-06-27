import numpy as np
import qregister as qr
import qoperator as qo
from collections import namedtuple

"""
This library implements a quantum circuit using efficient implementation of loacl unitaries using the  numpy einsum library
"""
def  sub_slice(array, dims, indices):
  """
   Returns a slice object that selects a subarray from a multidimensionalarray, with the values given by 
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
     
  return ind_list



#TODO check data types of inputs using isinstance function

Circ_sec = namedtuple('Circuit_Section', 'operator, qbitset, cntrl1, cntrl0') ## Single section of the circuit 
######################################################################################################
def apply_circ_sec(sec,phi):
  """
   To apply a given circuit section to a given numpy array reshaped to the shape (2,2,2...)
   Inputs: 
       sec : An Circ_sec instance
       phi : numpy array of the shape (2,2,2...)
       
   Output: 
        phi_out: numpy array after application of the circuit section
   
  """
  n = phi.ndim
  cntrl0 = sec.cntrl0
  cntrl1 = sec.cntrl1
  cntrls = [0]*len(cntrl0) + [1]*len(cntrl1)
  phi_sub = phi[sub_slice(phi, cntrl0 +cntrl1, cntrls)]
 
  #XXX np.eisum doesnt work if dimensions are more than 26
  k = len(sec.qbitset)
  U  = sec.operator.reshape([2]*(2*k))
  ind = sec.qbitset
  phi_ind = range(n)
  u_ind = range(n,n+k) + [phi_ind[i] for i in ind]
  indout = range(n)
  for index, i in enumerate(ind):
    indout[i] = range(n,n+k)[index] 

  phi_out_sub = np.einsum(U,u_ind,phi_sub,phi_ind, indout) 
  phi_out = phi.copy()
  phi_out[sub_slice(phi_out, cntrl0 +cntrl1, cntrls)] = phi_out_sub	

  #phi_out_sub=  sub_slice(phi_out, cntrl0 +cntrl1, cntrls)
  #phi_out_slice = phi_out_sub
  return phi_out
   

###########################################################################################################

class Qcircuit(object):

  def __init__(self,nbits):
    """
    nbits  gives the number of qubits in the circuit 
    """	
    self.nbits = nbits
    self.oper_list = []

  def insert_operator(self,qop, qbitset, cntrl1 = [], cntrl0 = []):
    """
     Add an operator to the circuit.
     qop - numpy array , numpy matrix or Qoperator, for the operator instance that we need to add
     qbitset - list of qubits on which qop acts
     cntrl1 - list of qubits from which we control qop (black dot)
     cntrl0 - list of qubits from which we control  qop (white dot)
       
    """


    if isinstance(qop,qo.Qoperator):
      op = np.asarray(qop.matrix) 
    elif isinstance(qop, np.matrix):
      op = np.asarray(qop) 
    else:
      op = qop   
    assert (isinstance(op, np.ndarray)), "qop must one of the three types specified in the documentation"   
    sec = Circ_sec(op, qbitset, cntrl1, cntrl0)
    self.oper_list.append(sec)
    
#########################################################################################  

  def inv(self):
    self.oper_list = self.oper_list[::-1]
    return




#########################################################################################  

  def evaluate(self, qreg, from_to = slice(None,)):
    """
     Find the effect of the circuit on the Qregister instance qreg
     from_to - Instead of evaluating the whole circuit we can evalute  parts of it
               eg: if from_to = slice(2,4), we evaluate the sub circuit given by the sections 2,3  on qreg.
               

    """	
   #XXX np.eisum doesnt work id dimensions are more than 26

   #XXX Proper errors and warnings
    n = self.nbits
    assert (isinstance(qreg, qr.Qreg) or isinstance(qreg, np.ndarray)), "Input register or array"
    
    if isinstance(qreg, qr.Qreg): 
      assert ( self.nbits == qreg.nbits), "Circuit and register must have same number of qubits"
      phi = qreg.array.reshape([2]*n)
      
    elif isinstance(qreg, np.ndarray):     
      assert (qreg.size == 2**n), "Circuit and register must have same number of qubits"
      phi = qreg.reshape([2]*n)  
    
    phi_out = np.zeros(phi.shape, dtype = np.complex128)
    phi_out[:] = phi[:]
    for op in self.oper_list[from_to]:
       phi_out = apply_circ_sec(op,phi_out)
       
    return qr.Qreg(n,phi_out)
    
############################################################################################    
  def __mul__(self, qreg):
  #XXX Dosent work with arrays
    n = self.nbits
    assert (isinstance(qreg, qr.Qreg) or isinstance(qreg, np.ndarray)), "Input register or array"
    
    if isinstance(qreg, qr.Qreg): 
      assert ( self.nbits == qreg.nbits), "Circuit and register must have same number of qubits"
      phi = qreg.array.reshape([2]*n)
      
    elif isinstance(qreg, np.ndarray):     
      assert (qreg.size == 2**n), "Circuit and register must have same number of qubits"
      phi = qreg.reshape([2]*n)
        
    phi_out = np.zeros(phi.shape, dtype = np.complex128)
    phi_out[:] = phi[:]
    for op in self.oper_list:
       phi_out = apply_circ_sec(op,phi_out)
       
    return qr.Qreg(n,phi_out)
    
###################################################################
  def __rmul__(self, qreg):
  #XXX Dosent work with arrays
    n = self.nbits
    assert (isinstance(qreg, qr.Qreg) or isinstance(qreg, np.ndarray)), "Input register or array"
    
    if isinstance(qreg, qr.Qreg): 
      assert ( self.nbits == qreg.nbits), "Circuit and register must have same number of qubits"
      phi = qreg.array.reshape([2]*n)
      
    elif isinstance(qreg, np.ndarray):     
      assert (qreg.size == 2**n), "Circuit and register must have same number of qubits"
      phi = qreg.reshape([2]*n)
        
    phi_out = np.zeros(phi.shape, dtype = np.complex128)
    phi_out[:] = phi[:]
    for op in self.oper_list:
       phi_out = apply_circ_sec(op,phi_out)
       
    return qr.Qreg(n,phi_out)  


    
  
    
  def __add__(self, other):
    #TODO Check
    
    assert ( self.nbits == other.nbits), "Circuits must have same number of qubits"
    out_circ = Qcircuit( self.nbits)
    out_circ.oper_list += self.oper_list + other.oper_list
    return out_circ 
    
  def __iadd__(self, other):
     #TODO Check
    
    assert ( self.nbits == other.nbits), "Circuits must have same number of qubits"
    self.oper_list += other.oper_list
    return self   
     
      
      
     
