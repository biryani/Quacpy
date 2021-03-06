"""
Functions for hamiltonian simulation
"""
import qregister as qr
import qoperator as qo
import auxfun as aux
import numpy as np
import matplotlib.pyplot as plt


def identityop(nbits):
	N = np.power(2,nbits)
	return qo.Qoperator(nbits,np.asmatrix(np.eye(N)))

def wireswitch(wire1, wire2, nbits):
	"""
	Switch two wires (given by the wire1 and wire2 arguments ) in a n qubit circuit
	NUmbering starts from zero
	"""	

	N = np.power(2,nbits)
	w1 = np.power(2,wire1)
	w2 = np.power(2,wire2)
	M = np.asmatrix(np.zeros((N,N), dtype = complex))
	for i in range(N):
		ev = np.zeros(nbits,dtype = int)
		reg1 = qr.getbasis(nbits,i)
		binlist = np.array(list(bin(i)[2:]), dtype = int)
		ev[nbits-binlist.size:] = binlist	
		j = i - ev[wire1]*w1 - ev[wire2]*w2 + ev[wire1]*w2 + ev[wire2]*w1
		reg2 = qr.getbasis(nbits,j)
		M = M + outerprod(reg2,reg1)
	return qo.Qoperator(nbits,M)

def switchop(nxbits,nybits):
	"""
	Go from row major order to column major order by wire switching
	"""
	nbits = nxbits + nybits	
	N = np.power(2,nbits)
	arr = np.arange(nbits-1,-1,-1)
	powarr = pow(2,arr)
	M = np.asmatrix(np.zeros((N,N), dtype = complex))
	for i in range(N):
		ev = np.zeros(nbits,dtype = int)
		reg1 = qr.getbasis(nbits,i)
		binlist = np.array(list(bin(i)[2:]), dtype = int)
		ev[nbits-binlist.size:] = binlist
		newbin = np.zeros(nbits,dtype =int)
		#TODO CHECK CORRECTNESS OF SWAPPINGS
		newbin[nxbits:] = ev[:nybits]
		newbin[:nxbits] = ev[nybits:]
		j = np.inner(powarr,newbin)
		reg2 = qr.getbasis(nbits,j)
		M = M + qr.outerprod(reg2,reg1)
	return qo.Qoperator(nbits,M)
	
		
			
	
def cyperm_op(nbits):
	"""
	Constructing a single shift cyclic permuation operator
	"""
	if nbits == 1:
		return 	 qo.Qoperator(1,np.asmatrix([[0,1],[1,0]]))
	else :
		S = qo.Qoperator(1,np.asmatrix([[0,1],[1,0]]))
		N = np.power(2,nbits)
		M1 = np.asmatrix(np.zeros((N/2,N/2)))
		M1[0,0] = 1
		M2 = np.asmatrix(np.eye(N/2) - M1)
		control0 = qo.Qoperator(nbits-1,M1)
        	control1 = qo.Qoperator(nbits-1,M2)
		temp1 = qo.directopprod(S,control0)
        	temp2 = qo.directopprod(identityop(1),control1)
		U2 = qo.Qoperator(nbits, temp1.matrix+temp2.matrix)
		U1 = qo.directopprod(identityop(1), cyperm_op(nbits-1))
		U  = qo.Qoperator(nbits,U2.matrix*U1.matrix)
		return U
	


def expT_2order_1D(nbits,dt,hbar,c):
	"""
	Simulate exp(-i*dt*T/hbar)
	error = O(dt^2)
	"""
	M = c*np.asmatrix([[1,-1],[-1,1]])
	expM = qo.Qoperator(1,qo.matpowerh(np.power(np.e,-1j), (dt/hbar)*M))
	expTeven = qo.directopprod(identityop(nbits-1),expM)
	P = cyperm_op(nbits)
	Pinv = qo.Qoperator(nbits,P.matrix.H)
	U = expTeven.matrix * P.matrix * expTeven.matrix * Pinv.matrix
	return qo.Qoperator(nbits,U)
def expH_3order_1D(nbits,dt,hbar,c,expV):
	"""
	Simulate exp(-i*dt*T/hbar)
	error = O(dt^3)
	"""
	M = c*np.asmatrix([[1,-1],[-1,1]])
	expM = qo.Qoperator(1,qo.matpowerh(np.power(np.e,-1j), (0.5*dt/hbar)*M))
	expTeven = qo.directopprod(identityop(nbits-1),expM)
	P = cyperm_op(nbits)
	Pinv = qo.Qoperator(nbits,P.matrix.H)
	U1 = expTeven.matrix * Pinv.matrix * expTeven.matrix * P.matrix
	U2 = P.matrix * expTeven.matrix * Pinv.matrix * expTeven.matrix  #XXX Check correctness of U1 and U2
	U = U1 * expV * U2
	return qo.Qoperator(nbits,U)
	
def expH_2order_2D(nxbits,nybits,dt,hbar,c,expV):
	"""
	Simulate exp(-i*dt*H/hbar)
	error = O(dt^2)
	"""
	nbits = nxbits + nybits
	M = c*np.asmatrix([[1,-1],[-1,1]])
	expM = qo.Qoperator(1,qo.matpowerh(np.power(np.e,-1j), (dt/hbar)*M))
	expTeven = qo.directopprod(identityop(nbits-1),expM)
	P = cyperm_op(nxbits)
	P1 = qo.directopprod(identityop(nybits),P)
	P1inv = qo.Qoperator(nbits,P1.matrix.H)
	expTodd = qo.Qoperator(nbits,P1inv.matrix * expTeven.matrix * P1.matrix)
	#TODO Find a linear algebra realization of the switching opeartor
	S = switchop(nxbits,nybits)
	expTup =   qo.Qoperator(nbits,S.matrix.H * expTeven.matrix * S.matrix)
	expTdown = qo.Qoperator(nbits,S.matrix.H * expTodd.matrix * S.matrix)
	U1 = expTeven.matrix * expTodd.matrix * expTup.matrix * expTdown.matrix
	U = U1 * expV 
	return qo.Qoperator(nbits, U)	
	

def expH_3order_2D(nxbits,nybits,dt,hbar,c,expV):
	"""
	Simulate exp(-i*dt*H/hbar)
	error = O(dt^3)
	"""
	nbits = nxbits + nybits
	M = c*np.asmatrix([[1,-1],[-1,1]])
	expM = qo.Qoperator(1,qo.matpowerh(np.power(np.e,-1j), (0.5*dt/hbar)*M))
	expTeven = qo.directopprod(identityop(nbits-1),expM)
	P = cyperm_op(nxbits)
	P1 = qo.directopprod(identityop(nybits),P)
	P1inv = qo.Qoperator(nbits,P1.matrix.H)
	expTodd = qo.Qoperator(nbits,P1inv.matrix * expTeven.matrix * P1.matrix)
	#TODO Find a linear algebra realization of the switching opeartor
	S = switchop(nxbits,nybits)
	expTup =   qo.Qoperator(nbits,S.matrix.H * expTeven.matrix * S.matrix)
	expTdown = qo.Qoperator(nbits,S.matrix.H * expTodd.matrix * S.matrix)
	U1 = expTeven.matrix * expTodd.matrix * expTup.matrix * expTdown.matrix
	U2 = expTdown.matrix * expTup.matrix * expTodd.matrix * expTeven.matrix
	U = U1 * expV * U2
	return qo.Qoperator(nbits, U)	



