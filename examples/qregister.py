#module for definitions of registers and related operations like innner product etc
import numpy as np
import auxfun as aux
####################################################################
class Qbasis(object):
	"""Set of basis vectors spanning the Hiblert space
							"""
	def __init__(self,nbits):
		self.size = np.power(2,nbits)
		self.set  = np.empty((self.size,self.size),dtype = complex)

	def setstd(self):
		self.set = np.identity(self.size, dtype = complex)
		return 
		

####################################################################
class Qreg(object):
	"""Class for an n-qubit quantum register
	contains : |qreg>						
	"""
	def __init__(self,nbits,array):
		self.nbits = nbits
		self.size = np.size(array)
		self.basis = Qbasis(nbits) 
		self.basis.setstd()##standard basis used by default
		if np.size(array) > np.power(2,nbits):
			print "Error: Size mismatch - make sure that the size of the array matches with the size of the Hilbert space"
			
		else:
			self.array = array/aux.qregnorm(array)
		

	def getamp(self,i):
		return self.array[i]
			
##########################################################################

def getbasis(nbits,i):
	"""
	Computes the basis ith basis vector and stores it in a nbit qbit register
	returns: |i>
	"""
	arr = np.zeros(np.power(2,nbits),dtype = complex)
	arr[i] = 1.0 
	return Qreg(nbits,arr)
	


def innerprod(qreg1,qreg2):
	"""
	inner product of two quantum registers
	returns: <qreg1|qreg2>			
	 """	
	if qreg1.size != qreg2.size:
		print "Size mismatch"
		return
	else: 
		return np.vdot(qre1.array,qreg2.array)

def outerprod(qreg1, qreg2):
	"""
	Outer product of two quantum registers 
	returns: |qreg1><qreg2|
	Note: This returns a  numpy martix, not a Qoperator
	"""
	#TODO : SHOULD I COMPLEX CONJUGATE THE SECOND ARRAY?
	#TODO : LOOKS CORRECT, BUT CHECK IT ANYWAY
	return np.matrix(np.outer(qreg1.array,np.conjugate(qreg2.array)))

def directprod(qreg1,qreg2):
	"""Direct product of two registers stored in a new register. 
	The array so formed is a multidimensional array so as to easily access component parts
	returns: |qreg1>|qreg2>"""
	arr1 = np.reshape(qreg1.array,np.size(qreg1.array))
	print arr1
	arr2 = np.reshape(qreg2.array,np.size(qreg2.array))
	outarray = np.kron(arr1,arr2)
	##TODO : INCLUDE BASIS COMPUATIONS, AS OF NOW EVERYTHING IS STD BASIS
	##TODO : CHECK CORRECTNESS OF THE SHAPE COMPUTATIONS AND BASIS COMPUTATIONS
	outshape = qreg1.array.shape+qreg2.array.shape
	outarray = np.reshape(outarray,outshape)
	qregout = Qreg(qreg1.nbits + qreg2.nbits,outarray)
	return qregout
###############################################################################
			

