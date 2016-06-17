import numpy as np
import qregister as qr
import qoperator as qo
from collections import namedtuple

"""
This library implements a quantum circuit using efficient implementation of loacl unitaries using the  numpy einsum library
"""
#TODO check data types of inputs using isinstance function

Circ_sec = namedtuple('Circuit_Section', 'operator, qbitset, cntrl1, cntrl0') ## Single section of the circuit

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
     qop - Qoperaotr instance that we need to add
     qbitset - list of qubits on which qop acts
     cntrl1 - list of qubits from which we control qop (black dot)
     cntrl0 - list of qubits from which we contrl  qop (white dot)
     
    """
    sec = Circ_sec(qop, qbitset, cntrl1, cntr0)
    self.oper_list.append(sec)

  def evaluate(self, qreg, from_to = None):
    """
     Find the effect of the circuit on the Qregister instance qreg
     from_to - Instead of evaluating the whole circuit we can evalute a parts of it
               eg: if from_to = [2,3,4], we evaluate the sub circuit given by the sections 2,3 and 4 on qreg.
               

    """	
   #TODO Figure out a way to implement control operations efficiently
   #XXX np.eisum doesnt work id dimensions are more than 26
   #TODO Implement checks eg: qreg.nbits and self.nbits should match
   #XXX Write testers for this
    if from_to == None:
       from_to = range(self.seclist.__len__()) 
       n = self.nbits
       phi = qreg.array.reshape([2]*nbits)
       phi_out = phi ##XXX  Copy of ref?
    for op in self.oper_list[from_to]:
       k = op(qbitset).__len__()
       U  = op(operator).reshape([2]*(2*k))
       ind = op(qbitset)
       #TODO Match the qubit indices given in ind and write the einsum function  
       phi_ind = range(n)
       u_ind = range(n,n+k) + phi_ind[ind]
       phi_out = np.einsum(U,u_ind,phi_out,phi_ind) 
       
    return Qreg(n,phi_out)
  def __mult__(self, qreg):
    return evaluate(self, qreg) 
      
     
