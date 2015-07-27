"""
Making faster functions for simulation by using spare matrix computations and numpy vectorization
Faster version of the simulation.py library
2D case checked at test5 in test_bed folder. Comes out correct
"""
import numpy as np





def Apply_expTeven(phi, c, nbits, hbar, dt):
	#Compared and checked : Works perfectly
	#TODO Make sure that hte vector has even number of entries

	a = np.zeros(phi[::2].shape, dtype = np.complex128)
	#M = c*np.asmatrix([[1,-1],[-1,1]])
	#expM = qo.matpowerh(np.power(np.e,-1j), (dt/hbar)*M)
	alpha = np.exp(-1j*dt*2*c/hbar)
	expM = 0.5*np.asmatrix([[1.0+alpha, 1.0-alpha],[1.0-alpha, 1.0+alpha]], dtype = np.complex128)
	## Multiplication with the Teven laplacian
	##Using swapping
	phi1 = np.zeros(phi.shape, dtype = np.complex128)
	phi1[:] = phi[:]
	a[:] = phi[::2]
	phi[::2] = phi[1::2]
	phi[1::2] = a[:]
	phi1 = phi1*expM[0,0]
	phi  = phi*expM[0,1]
	return phi1 + phi


	
def Apply_cyclicpermut_2D( phi, Nx, Ny):
	#Checked an tested: working
	phi = phi.reshape(Nx,Ny)
	phi = np.apply_along_axis(np.roll,1,phi,1)
	phi = phi.flatten().reshape(Nx*Ny ,1)
	return phi

def Apply_cyclicpermut_H_2D(phi, Nx, Ny):
	#Checked an tested: working
	phi = phi.reshape(Nx,Ny)
	phi = np.apply_along_axis(np.roll,1,phi,-1)
	phi = phi.flatten().reshape(Nx*Ny ,1)
	return phi
def Apply_rowcolswap(phi, Nx, Ny):
	phi = phi.reshape(Nx,Ny)
	phi = phi.transpose()
	phi = phi.flatten()
	phi = phi.flatten().reshape(Nx*Ny ,1)
	return 	phi

def Apply_rowcolswap_H(phi,Nx,Ny):
	phi = phi.reshape(Ny,Nx)
	phi = phi.transpose()
	phi = phi.flatten()
	return phi


def Fastsimulate_1D_2order(phi, c, nbits, expV, hbar,dt):
	phi = Apply_expTeven(phi, c, nbits, hbar, dt)
	phi = np.roll(phi,1)
	phi = Apply_expTeven(phi, c, nbits, hbar, dt)
	phi = np.roll(phi,-1)
	phi = phi*expV
	return phi

def Fastsimulate_1D_3order(phi, c,nbits, expV, hbar,dt):
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = np.roll(phi,1)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = np.roll(phi,-1)
	phi = phi*expV
	phi = np.roll(phi,1)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = np.roll(phi,-1)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	return phi


def Fastsimulate_2D_2order(phi, c, nbits, expV, Nx, Ny, hbar,dt):

	phi = Apply_expTeven(phi, c, nbits, hbar, dt)
	phi = Apply_cyclicpermut_2D(phi,Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, dt)
	phi = Apply_cyclicpermut_H_2D(phi,Nx,Ny)
	phi = Apply_rowcolswap(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, dt)
	phi = Apply_cyclicpermut_2D(phi,Nx,Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, dt)
	phi = Apply_cyclicpermut_H_2D(phi,Nx,Ny)
	phi = Apply_rowcolswap_H(phi, Nx, Ny)
	phi = phi*expV
	return phi

def Fastsimulate_2D_3order(phi, c, nbits, expV, Nx, Ny, hbar,dt):
	#Tested against the simulation library and brute force exponentiations. Working, no error found. 

	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = Apply_cyclicpermut_2D(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = Apply_cyclicpermut_H_2D(phi, Nx, Ny)
	phi = Apply_rowcolswap(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = Apply_cyclicpermut_2D(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = Apply_cyclicpermut_H_2D(phi, Nx, Ny)
	phi = Apply_rowcolswap_H(phi, Nx, Ny)
	
	phi = phi*expV
	
	phi = Apply_rowcolswap(phi, Nx, Ny)
	phi = Apply_cyclicpermut_2D(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = Apply_cyclicpermut_H_2D(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = Apply_rowcolswap_H(phi, Nx, Ny)
	phi = Apply_cyclicpermut_2D(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)
	phi = Apply_cyclicpermut_H_2D(phi, Nx, Ny)
	phi = Apply_expTeven(phi, c, nbits, hbar, 0.5*dt)

		
	return phi


	







