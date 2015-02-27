import numpy as np
import matplotlib.pyplot as plt
import sys

class ith_constants:
	def __init__(self, k, L, Lw):
		if (not len(k) == 4) or (not len(L) == 4):
			raise BaseException("wrong input length!")
		self.k = k
		self.L = L
		self.Lw = Lw
	

constants_dict = {
	'a': ith_constants(k=(1.,2.,3.,4.), L=(1.,1.,1.,1.), Lw=4.),
	'b': ith_constants(k=(1.,2.,3.,4.), L=(1.,1.,1.,1.), Lw=10.),
	'c': ith_constants(k=(1.,1.,1.,1.), L=(2.,2.,1.,1.), Lw=4.),
	'd': ith_constants(k=(1.,1.,1.,0.), L=(2.,2.,1.,1.), Lw=4.),
	'e': ith_constants(k=(0.,1.,1.,0.), L=(2.,2.,1.,1.), Lw=4.)
}


# Make A and B matrices.
def make_A_and_B_matrices(k, L, Lw):
	A = np.array([[-k[0] -k[1], k[1], 0.],
		[k[1], -k[1] -k[2], k[2]],
		[0., k[2], -k[2] -k[3]]])

	B = np.array([ -k[0]*L[0] + k[1]*L[1],
		-k[1] * L[1] + k[2]*L[2],
		-k[2] * L[2]+k[3]*(L[3]-Lw)])

	return A, B


# Solve for x in A * x = B.
def find_x(A, B):
	if np.linalg.cond(A) > 1/sys.float_info.epsilon:
		raise BaseException("Ill-conditioned A-matrix!: A={}".format(A))	

	if np.linalg.det(A) == 0:
		raise BaseException('A is not invertible! Det=0. A={}'.format(A))

	return np.dot(np.linalg.inv(A), B)

def make_plot(Lw_array, F_rw_m, F_rw_t, k, L):
	fig, ax = plt.subplots()
	ax.grid()
	ax.set_xlabel('$L_w$, the distance between walls (meters).')
	ax.set_ylabel('Force on Right Wall (N)')
	ax.plot(Lw_array, F_rw_m,  label='F_rw from matrix inversion')
	ax.plot(Lw_array, F_rw_t, '--', label='F_rw predicted')
	plt.title('$k$={}\n $L$={} '.format(k, L))
	plt.legend()
	plt.show()

def do_part_1():
	for part, constants in sorted(constants_dict.iteritems()):
		k, L, Lw = constants.k, constants.L, constants.Lw
		A, B = make_A_and_B_matrices(k, L, Lw)
		print 'part ({}): k={}, L={}, Lw={}'.format(part, k, L, Lw)
		x = find_x(A, B)
		print '\t-> x = {}\n'.format(x)
	
def do_part_2():
	for part in ['a']: # you can add other parts to this list for more cases.
		constants = constants_dict[part]
		k, L, Lw = constants.k, constants.L, constants.Lw
		Force_right_wall_measured = []
		Force_right_wall_theoretical = []
		k_0 = 1/sum([1/k_i for k_i in k]) 
		L_0 = sum(L)

		Lw_array = np.linspace(0, 15, 100)
		for Lw_i in Lw_array:
			A, B = make_A_and_B_matrices(k, L, Lw_i)
			X = find_x(A, B)
			Force_right_wall_measured.append(-k[3] * (Lw_i - X[2] - L[3]))
			Force_right_wall_theoretical.append(-k_0 * (Lw_i - L_0))

		make_plot(Lw_array, np.array(Force_right_wall_measured), 
			np.array(Force_right_wall_theoretical), k, L)



def main():

	# do_part_1() # this should throw an non-invertible error.
	do_part_2()	

main()
