import numpy as np
import qregister as qr
import qoperator as qo
from collections import namedtuple

"""
This library implements a quantum circuit using efficient implementation of loacl unitaries using the  numpy einsum library
"""
def  sub_slicer(array, dims, indices):
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



#TODO check data types of inputs using isinstance function

Circ_sec = namedtuple('Circuit_Section', 'operator, qbitset, cntrl1, cntrl0') ## Single section of the circuit
def apply_circ_sec(sec,phi):
  """
   To apply a given circuit section to a given numpy array reshaped to the shape (2,2,2...)
   Inputs: 
       sec : An Circ_sec instance
       phi : numpy array of the shape (2,2,2...)
       
   Output: 
        phi_out: numpy array after application of the circuit section
   
  """
  #TODO Implement control operations
  #XXX np.eisum doesnt work if dimensions are more than 26
  k = sec(qbitset).__len__()
  U  = sec(operator).reshape([2]*(2*k))
  ind = sec(qbitset)
  phi_ind = range(n)
  u_ind = range(n,n+k) + phi_ind[ind]
  phi_out = np.einsum(U,u_ind,phi,phi_ind) 
  return phi_out 
   



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
     qop - Matrix instance that we need to add
     qbitset - list of qubits on which qop acts
     cntrl1 - list of qubits from which we control qop (black dot)
     cntrl0 - list of qubits from which we contrl  qop (white dot)
       
    """
    #TODO The user adds a matrix and not a Qoperator instance. Add code to convert the operator to a Qoperaor instance
    #TODO Check if the size of the operator matches with the size of the  qubit set
    sec = Circ_sec(qop, qbitset, cntrl1, cntr0)
    self.oper_list.append(sec)

  def evaluate(self, qreg, from_to = None):
    """
     Find the effect of the circuit on the Qregister instance qreg
     from_to - Instead of evaluating the whole circuit we can evalute a parts of it
               eg: if from_to = [2,3,4], we evaluate the sub circuit given by the sections 2,3 and 4 on qreg.
               

    """	
   #XXX np.eisum doesnt work id dimensions are more than 26
   #TODO Implement checks eg: qreg.nbits and self.nbits should match
   #XXX Write testers for this
    if from_to == None:
       from_to = range(self.seclist.__len__()) 
       n = self.nbits
       phi = qreg.array.reshape([2]*nbits)
       phi_out = np.zeros(phi.shape)
       phi_out[:] = phi[:]
    for op in self.oper_list[from_to]:
       phi_out = apply_circ_sec(op,phi_out)
       
    return Qreg(n,phi_out)
  def __mult__(self, qreg):
    return evaluate(self, qreg) 
      
     
