import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def relax(N_gridpoints, L, boundary_vectors, cage_points=[]):

	# Initialize parameters.
	N_gridpoints = N_gridpoints
	L = L
	h_gridspacing = L / (N_gridpoints - 1.)
	x = (np.arange(N_gridpoints) - 1) * h_gridspacing
	y = np.copy(x)

	# Select over-relaxation factor.
	omega_optimal = 2./(1. + np.sin(np.pi / N_gridpoints))
	omega_desired = omega_optimal


	# Set initial guess as first term in speration of variables soln.
	#phi_0 = 1 # potential at y=L.
	#coeff = (phi_0 * 4.) / (np.pi * np.sinh(np.pi))
	phi = np.zeros([N_gridpoints, N_gridpoints])
	#for i in np.arange(N_gridpoints):
	#	for j in np.arange(N_gridpoints):
	#		phi[i, j] = coeff * np.sin(np.pi * x[i] / L) * np.sinh(np.pi * y[j] / L)

	# Set boundary conditions.
	phi[0] = boundary_vectors['x=0']
	phi[N_gridpoints-1] = boundary_vectors['x=L']
	phi.T[0] = boundary_vectors['y=0']
	phi.T[N_gridpoints-1] = boundary_vectors['y=L']

	# Loop until desired fractional change per iteration is obtained.
	max_iter = N_gridpoints**2
	change_desired = 1e-10
	nth_fractional_change = []

	for iteration in range(max_iter):
		change_sum = 0.
		for i in np.arange(1, N_gridpoints-1):
			for j in np.arange(1, N_gridpoints-1):
				if (i, j) in cage_points:
					phi[i, j] = 0
				else:
					phi_temporary = 0.25 * omega_desired * (phi[i+1, j] + phi[i-1, j] 
				 		+ phi[i, j-1] + phi[i, j+1]) + (1 - omega_desired) * phi[i, j]

					change_sum += np.abs(1 - (phi[i, j] / phi_temporary))
					phi[i, j] = phi_temporary

		# Check if fractional change is small enough to halt the iteration.
		nth_fractional_change.append(change_sum /((N_gridpoints-2)**2))
		if nth_fractional_change[-1] < change_desired:
			break

	return phi, nth_fractional_change

	
def make_3d_plot(phi, title):
	
	N = len(phi[0])
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X, Y = np.meshgrid(np.arange(N), np.arange(N))
	cs = max(2, N/40)
	rs = max(2, N/40)
	print 'shape(X): {}, shape(Y): {}, shape(phi): {}'.format(np.shape(X), np.shape(Y),
		 np.shape(phi))
	ax.plot_surface(X,Y,phi,cstride=cs,rstride=rs,cmap='jet')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	ax.set_zlabel('$\Phi(x,y)$')
	plt.title(title)
	plt.tight_layout()
	plt.show()

# Each cage will generate two files. Those files are named the following.
def make_data_filename(basename, phi_or_change=None):
	filename = 'XXX'
	if phi_or_change == 'phi':
		filename = '{}_phi.csv'.format(basename)
	elif phi_or_change == 'change':
		filename = '{}_change.csv'.format(basename)
	else:
		raise BaseException('invalid!')
	return filename

def compute_different_phi_save_to_file():
	N_gridpoints_per_side = 60
	boundary_vectors_dict = {
		'x=0': np.zeros(N_gridpoints_per_side),
		'x=L': np.ones(N_gridpoints_per_side) * 100,
		'y=0': np.linspace(0, 100, N_gridpoints_per_side),
		'y=L': np.linspace(0, 100, N_gridpoints_per_side)
		}
	runs_dictionary = {
		'no_cage': [],
		'cage_a': [(19, 19), (29, 19), (39, 19), (19, 29), (19, 39), (29, 39), (39, 29), (39, 39)],
		'cage_b': [(19, 19), (19, 39), (39, 19), (39, 39)],
		'cage_c': [(19,29), (29,19), (39,29), (29, 39)]
	}
	for cagekey, cage in runs_dictionary.iteritems():
		print 'computing phi for {} ...'.format(cagekey)
		phi_final, nth_fractional_change = relax(N_gridpoints=60, L=1.,
		boundary_vectors=boundary_vectors_dict, cage_points=cage)
		np.savetxt(fname=make_data_filename(cagekey, 'phi'), X=phi_final)
		np.savetxt(fname=make_data_filename(cagekey, 'change'), X=nth_fractional_change)
		print '... saved'

	
def main():
	N_gridpoints_per_side = 60
	boundary_vectors_dict = {
		'x=0': np.zeros(N_gridpoints_per_side),
		'x=L': np.ones(N_gridpoints_per_side) * 100,
		'y=0': np.linspace(0, 100, N_gridpoints_per_side),
		'y=L': np.linspace(0, 100, N_gridpoints_per_side)
		}
	cage_a = [(19, 19), (29, 19), (39, 19), (19, 29), (19, 39), (29, 39), (39, 29), (39, 39)]
	cage_b = [(19, 19), (19, 39), (39, 19), (39, 39)]
	cage_c = [(19,29), (29,19), (39,29), (29, 39)]
	phi_final, nth_fractional_change = relax(N_gridpoints=N_gridpoints_per_side, L=1.,
		boundary_vectors=boundary_vectors_dict, cage_points=cage_a )
	print phi_final
	make_3d_plot(phi_final, 'plot')
	return phi_final, nth_fractional_change


if __name__ == '__main__':
	compute_different_phi_save_to_file()
