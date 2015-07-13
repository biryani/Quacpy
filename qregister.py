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
		if np.size(array) > np.power(2,nbits):
			print "Error: Size mismatch - make sure that the size of the array matches with the size of the Hilbert space"
			
		else:
			self.array = array/aux.qregnorm(array)
			self.regshape = array.shape
			self.array = np.reshape(self.array,(self.array.size,1))
		

	def reshape(self):
		arr = np.reshape(self.regshape,self.array)
		return Qreg(self.nbits,self.array)
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
	outshape = qreg1.regshape+qreg2.regshape
	outarray = np.reshape(outarray,outshape)
	qregout = Qreg(qreg1.nbits + qreg2.nbits,outarray)
	return qregout
###############################################################################
def schdecomp(qreg):
	"""
	Schmidt decomposition of a pure state of a bipartite system
	|qreg> = c|u1>|u2>
	Returns |u1>|u2>
	Note: Check the purity of the system by using the schcoeff function so as to make sure that there is no entanglement
	Note: ENTANGLED INPUT WILL GIVE DUBIOUS RESULTS
	"""
	a = np.reshape(qreg.array,qreg.regshape)	
	##FIXME CHECK CORRECTNESS
	## CHECKED: LOOKS CORRECT	
	n1 = qreg.regshape[0]
	n2 = qreg.regshape[1]
	
	if len(qreg.regshape) != 2:
		print "Error: register is not bipartite"
		return
	else:
		U, s, V = np.linalg.svd(a)
		i1 =  U[:,0]
		i2 =  V[0,:]
		nb1 = int(np.ceil(np.log2(n1)))	
		nb2 = int(np.ceil(np.log2(n2)))	
		return Qreg(nb1,i1), Qreg(nb2,i2)	
####################################################################################	
def schdecomp_multi(qreg,split):
	"""
	Schmidt decomposition of a pure state of a multipartite system into a two sperate registers whose size is 
	given by the split tuple. 
	|qreg> = c|u1>|u2>
	Returns |u1>|u2>
	"""
	##XXX I HAVE NO FAITH IN THE CORRECTNESS OF THIS FUNCTION
	sh = np.power(2,split)
	a = np.reshape(qreg.array,sh)	
	n1 = sh[0]
	n2 = sh[1]
	U, s, V = np.linalg.svd(a)
	i1 =  U[:,0]
	i2 =  V[0,:]
	nb1 = int(np.ceil(np.log2(n1)))	
	nb2 = int(np.ceil(np.log2(n2)))	
	return Qreg(nb1,i1), Qreg(nb2,i2)				

