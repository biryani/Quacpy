"""
auxillay functions used in the  QuacPy module
"""
import numpy as np
import matplotlib.pyplot as plt
def qregnorm(array):
	"""
	Norm of an array in the register
	"""
	a = np.reshape(array,np.size(array))
	return np.linalg.norm(a)

def qplotreg(qreg,flag):
	"""Plots the amplitude values stored in the register
		if flag is set to abs -> plots absolute values
		if flag is set to phase -> plots phases
	"""
	a = np.reshape(qreg.array, qreg.array.size)
	if flag == "abs":
		a = np.absolute(a)
	if flag == "phase":
		a = np.angle(a)
	
	plt.plot(a)
	plt.ylabel(flag)
	plt.xlabel("index")
	plt.show()
	return

def schmidtcoeff(qreg):
	"""
	Schmidt coefficients of a bipartite system
	"""
	if len(qreg.array.shape) != 2:
		print "Error: register is not bipartite"
		return
	else:
		U, s, V = np.linalg.svd(qreg.array)
		return s	
		
	
	
