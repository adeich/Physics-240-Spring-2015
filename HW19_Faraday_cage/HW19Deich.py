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

	
def make_and_save_plots(phi, change_vec, title):
	print 'plotting {}.'.format(title)	


	N = len(phi[0])
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	X, Y = np.meshgrid(np.arange(N), np.arange(N))
	cs = max(2, N/40)
	rs = max(2, N/40)
	ax.plot_surface(X,Y,phi,cstride=cs,rstride=rs,cmap='jet')
	plt.xlabel('$x$')
	plt.ylabel('$y$')
	ax.set_zlabel('$\Phi(x,y)$')
	plt.title(title)
	plt.tight_layout()
	#plt.show()
	plt.savefig('{}_3d.png'.format(title))

	fig2 = plt.figure()
	plt.plot(np.arange(len(change_vec)), change_vec)
	plt.grid(True)
	plt.xlabel('Iteration $N$')
	plt.ylabel('Fractionial change')
	plt.title('{} Fractional change per iteration'.format(title))
	plt.yscale('log')
	plt.savefig('{}_2d_change.png'.format(title))

	fig3 = plt.figure()
	plt.plot(np.arange(len(phi[0])), phi[29])
	plt.xlabel('$x$')
	plt.ylabel('$\Phi(x,y)|_{y=29}$')
	plt.title('{} Potential along $y=29$'.format(title))
	plt.savefig('{}_2d_crosssection.png'.format(title))


def make_plot_compare_all(phi_list, change_list, title_list):
	
	fig2 = plt.figure()
	for change_vec, title in zip(change_list, title_list):
		plt.plot(np.arange(len(change_vec)), change_vec, label=title)
	plt.grid(True)
	plt.xlabel('Iteration $N$')
	plt.ylabel('Fractionial change')
	plt.title('{} Fractional change per iteration'.format(title))
	plt.yscale('log')
	plt.legend()
	plt.savefig('All_2d_change.png')

	fig3 = plt.figure()
	for phi, title in zip(phi_list, title_list):
		plt.plot(np.arange(len(phi[0])), phi[29], label=title)
	plt.xlabel('$x$')
	plt.ylabel('$\Phi(x,y)|_{y=29}$')
	plt.title('{} Potential along $y=29$'.format(title))
	plt.legend()
	plt.savefig('All_2d_crosssection.png')




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

def compute_different_phi_save_to_file(runs_dictionary):
	N_gridpoints_per_side = 60
	boundary_vectors_dict = {
		'x=0': np.zeros(N_gridpoints_per_side),
		'x=L': np.ones(N_gridpoints_per_side) * 100,
		'y=0': np.linspace(0, 100, N_gridpoints_per_side),
		'y=L': np.linspace(0, 100, N_gridpoints_per_side)
		}
	for cagekey, cage in runs_dictionary.iteritems():
		print 'computing phi for {} ...'.format(cagekey)
		phi_final, nth_fractional_change = relax(N_gridpoints=N_gridpoints_per_side, L=1.,
		boundary_vectors=boundary_vectors_dict, cage_points=cage)
		np.savetxt(fname=make_data_filename(cagekey, 'phi'), X=phi_final)
		np.savetxt(fname=make_data_filename(cagekey, 'change'), X=nth_fractional_change)
		print '... saved'


def load_files_and_make_plots(runs_dictionary):

	phi_list, change_list, title_list = [], [], []

	for cagekey, cage in runs_dictionary.iteritems():
		phi_filename = make_data_filename(cagekey, phi_or_change='phi')
		change_filename = make_data_filename(cagekey, phi_or_change='change')

		phi = np.loadtxt(phi_filename)
		change_vector = np.loadtxt(change_filename)

		phi_list.append(phi)
		change_list.append(change_vector)
		title_list.append(cagekey)

		# Make individual plots.
		make_and_save_plots(phi, change_vector, title=cagekey)

	# Make set of comparative plots.
	make_plot_compare_all(phi_list, change_list, title_list)

	
def main():

	runs_dictionary = {
		'no_cage': [],
		'cage_a': [(19, 19), (29, 19), (39, 19), (19, 29), (19, 39), (29, 39), (39, 29), (39, 39)],
		'cage_b': [(19, 19), (19, 39), (39, 19), (39, 39)],
		'cage_c': [(19,29), (29,19), (39,29), (29, 39)]
	}

	# Run this to compute phi in the first place.
	# Once it's run, you can comment it out.
	compute_different_phi_save_to_file(runs_dictionary)

	load_files_and_make_plots(runs_dictionary)


if __name__ == '__main__':
	main()
