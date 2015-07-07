import numpy as np
import auxfun as aux
import qregister as qr


def isunitary(A):
	"""
	Check if a martix is unitary
	"""
	E = A*(A.H)
	n = np.size(A,1)
	res = np.linalg.norm(E - np.identity(n))
	if res < 0.001:
		return True
	else:  	
		print "Not Unitary",res
		return False
 
def ishermitian(A):
	"""
	Check if a matrix is Hermitian
	"""	
	n = np.size(A,1)
	res = np.linalg.norm(A.H - A)
	if res < 0.001:
		return True
	else:  	
		print "Not Hermitian",res
		return False
#################################################################################################################################### 	

class Qoperator(object):
	"""Class for a unitary quantum operator
	contains: U or exp(iAt) or whatever...	
	"""
	def __init__(self,nbits,matrix):
		self.matrix = matrix
		self.size =  np.size(matrix,1)
		self.nbits = nbits
	 	self.basis = qr.Qbasis(nbits)
		self.basis.setstd()
		if self.size > np.power(2,nbits):
			print "Error: Size mismatch - make sure that the size of the array matches with the size of the Hilbert space"
		
		if  isunitary(matrix) == False:
			print "Warning: Matrix is not unitary"

	def invert(self):
		"""
		return the inverted operator 
		returns: U dagger (Hermitian conjugate)
		"""
		return Qoperator(self.nbits,self.matrix.H)

def addop(U1,U2):
	"""
	Add two unitary operators
	returns: U1 + U2
	"""
	if U1.nbits != U2.nbits:
		print "Error: size mismatch"		
		return
	else:
		return Qoperator(U1.nbits, U1.matrix + U2.matrix)
			

def operate(U,r):
	"""
	Apply the operator U on the register r
	returns U|r>
	"""
	if U.nbits != r.nbits:
		print "Error: size mismatch between operator and register"
		return
	else:	
		arr = np.ndarray.flatten(r.array)
		arr = np.asmatrix(arr, dtype = complex).T
		out = U.matrix*arr
		out = np.reshape(out,r.array.shape)
		out = np.asarray(out, dtype = complex)
		return qr.Qreg(U.nbits,out)
######################################################################################################################################

def matpowerh(a,A):
	"""
	returns a^A where A is Hermitian
	"""
	###TODO : CHECK CORRECTNESS
	w,V = np.linalg.eigh(A)
	w = np.power(a,w)
	W =  np.matrix(np.diag(w))
	return V*W*V.H
	


def evolution(H,c):
	"""
	returns: U = exp(jHc) if H is hermitian
	Note: Brute force. Computationally inefficient
	"""
	if ishermitian(H) == False:
		print "H is not Hermitian"
		return
	else:
		U = matpowerh(np.power(np.e,1j*c),H)
		return U
#################### Pauli Operators and other useful operators################################################################

def pauliX():
	X = np.matrix([[0,1],[1,0]])
	return Qoperator(1,X)
def pauliY():
	Y = np.matrix([[0,-1j],[1j,0]])
	return Qoperator(1,Y)
def pauliZ():
	Z = np.matrix([[1,0],[0,-1]])
	return Qoperator(1,Z)

def hadamard():
	"""
	The single qubit Hadamard operator
	"""
	H = np.matrix([[1,1],[1,-1]])/np.power(2,0.5)
	return Qoperator(1,H)

def groverreflector(n):
	"""
	The Grover reflector on an n dimensional Hilbert space
	returns: 2|u><u| - I
	where  u = 1/sqrt(n) \sum_{i = 0}^{N-1} |i>
	"""
	u = np.ones(n)
	u = u/np.power(n,0.5)
	G = 2*direct(u,u) - np.identity(n)
	nbits = np.ceil(np.log2(n))
	return Qoperator(nbits,G)

############################################Direct product of operators#################################################################

def directopprod(qop1,qop2):
	"""
	Direct product of two operators
	returns qop1 \otimes qop2
	"""
	outmat = np.kron(qop1.matrix,qop2.matrix)
	return Qoperator(qop1.nbits + qop2.nbits, outmat) 

def nbithadamard(nbits):
	"""
	Returns the Hadamard operator on n qubits
	Recursively uses the directopprod function
	"""	
	if nbits ==1:
		return hadamard()
	else:
		return directopprod(nbithadamard(nbits-1),hadamard())
	
	#TODO ASK ADP ABOUT CASES IN DIMENSIONS THAT ARE NOT POWERS OF 2.CHECK OUT SCIPY IMPLEMENTATION
	return
################################################ QFT ##########################################################################

def qftop(n):
	"""
	Returns the QFT operator of size n (n is not the number of q-bits, pass 2^nbits for that)
	Defined with a negtive sign in exponent as used in scipy. This is the opposite of what is given in Mike and Ike
	"""
	dft = np.fft.fft(np.identity(n))/np.sqrt(n)
	nbits  = np.ceil(np.log2(n))
	return Qoperator(nbits,np.asmatrix(dft))

def iqftop(n):
	"""
	Returns the inverse QFT operator of size n (n is not the number of bits, pass 2^nbits for that)
	Defined with a positive sign in exponent as used in scipy. This is the opposite of what is given in Mike and Ike
	"""
	dft = np.fft.ifft(np.identity(n))/np.sqrt(n)
	nbits  = np.ceil(np.log2(n))
	return Qoperator(nbits,np.asmatrix(dft))
	
	


	

	
		
	
		
	

