# schro.py : solve Schrodinger's equation for free-space wavepacket
#               using Crank--Nicolson method

from pylab import *
from numpy.linalg import inv


class constants():
	def __init__(self, L=None, N_x=30):
		self.m = 1.0
		self.hbar = 1.0
		if L:
			self.L = L
		self.k0 = 0.5
		self.sigma0 = self.L/10.
		self.x0 = 0.0
		self.p0 = self.hbar * self.k0
		self.N_x = N_x
		self.dx = self.L / self.N_x
		self.v0 = self.p0 / self.m
		self.T_cycle = self.L / self.v0
		self.dt = self.dx / self.v0
		self.N_t = int(self.T_cycle / self.dt)


def original_schro(gc, initial_x_function):
	m, hbar, L = gc.m, gc.hbar, gc.L
	k0, sigma0, x0 = gc.k0, L / 10.0, gc.x0
	p0 = hbar * k0
	v0 = gc.p0 / gc.m

	N_x = gc.N_x
	dx = gc.dx # gc.L / N_x
	dt = gc.dt #dx / v0
	#T_cycle = L / v0
	N_t = gc.N_t # int(T_cycle / dt)
	x = linspace(-L/2.0,L/2.0,N_x)
	IM = identity(N_x)

	# Hamiltonian: 
	H = -2.0 * IM + roll(IM,1,axis=1) + roll(IM,-1,axis=1)
	H *= -hbar**2 / (2.0 * m * dx**2)

	# Crank-Nicolson matrix:
	D_CN = dot(inv(IM + 0.5j*dt/hbar * H),(IM - 0.5j*dt/hbar * H))

	# initial conditions for wavefunction :
	# psi0 = exp(1.0j*k0*x) * exp(-(x-x0)**2 / 2.0 / sigma0**2) / sqrt(sqrt(pi) * sigma0)
	psi0 = initial_x_function(gc, x)
	psi = copy(psi0)

	# probability density :
	P = conj(psi0)*psi0

	for n in range(N_t) :
		psi = dot(D_CN, psi)
		P = vstack((P,conj(psi)*psi))

	return x, psi0, psi, P

def make_plots(x, psi0, psi, P, global_constants):
	L = global_constants.L
	N_x = global_constants.N_x
	N_t = global_constants.N_t	
	rcParams.update({'font.size': 18})
	fig1 = figure(figsize=(10,7))
	plot(x,real(psi0),'k--',label='real, initial')
	plot(x,imag(psi0),'r--',label='imaginary, initial')
	plot(x,real(psi),'k-',label='real, final')
	plot(x,imag(psi),'r-',label='imaginary, final')
	xlabel('$x$')
	ylabel('$\psi(x,t)$')
	legend(loc='best',fontsize=13)
	grid('on')
	xlim(-L/2.0,L/2.0)
	title('wavepacket propagation, $N_x=%d$' % N_x)
	savefig('schro_fig1.png')

	fig2 = figure(figsize=(10,7))
	plot_steps = range(0,N_t,N_t/10) # sequence of 10 time-points
	plot(x,P[0],'k--',label='initial')
	for n in plot_steps[1:] :
		plot(x,P[n])
	plot(x,P[-1],'k-',label='final')
	xlabel('$x$')
	ylabel('$P(x,t)$')
	xlim(-L/2.0,L/2.0)
	legend(loc='best',fontsize=13)
	grid('on')
	title('wavepacket probability density, $N_x=%d$' % N_x)
	savefig('schro_fig2.png')

	show()


def main():
	global_constants = constants(L=100.)

	# define the initial wave distribution at t=0.
	initial_x_function1 = lambda gc, x: exp(1.0j*gc.k0*x) * exp(-(x-gc.x0)**2 / 2.0 / gc.sigma0**2) / sqrt(sqrt(pi) * gc.sigma0)
	x, psi0, psi, P = original_schro(global_constants, initial_x_function1)
	make_plots(x, psi0, psi, P, global_constants)


main()
