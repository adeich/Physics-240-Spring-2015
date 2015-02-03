import numpy as np
import matplotlib.pyplot as plt

# Absolute fractional error (takes 2 numbers; returns 1 number).
def compute_abs_frac_error(true_number, approx_number):
	if true_number is None or approx_number is None:
		raise BaseException("got a None for x:{}, N:{}".format(true_number, approx_number))
	return np.abs((approx_number - true_number)/true_number)


# Computes Taylor approximation of e**(x).
def approximate_exp(x, N):
	total = np.sum([(x**i)/(np.math.factorial(i)) for i in np.arange(N)])
	return total


# Computes the array for the abs-frac-error for N=60 and specified x.  
def create_graphing_data(x):
	N = 60

	# an array of incremental integer N values.
	#independent_axis = np.arange(1, N + 1)
	
	#x_clones_array = np.empty(N); x_clones_array.fill(x)

	# an array of absolute fractional error for each N value.
	#dependent_axis = compute_abs_frac_error(np.exp(x_clones_array), 
	#	approximate_exp(x_clones_array, independent_axis))
	
	# No vectorization here, as I couldn't get it to work.
	# Sadly, a plain old Python for-loop.
	independent_axis = np.linspace(1, N, N)
	dependent_axis = []
	for n in independent_axis:
		dependent_axis.append(compute_abs_frac_error(np.exp(x), approximate_exp(x, n)))

	return (np.array(independent_axis), np.array(dependent_axis))
	
# Create 2x2 grid containing 4 plots (one for each x-value).
def plot_2by2_grid(four_x_values_list):

	plt.suptitle('Approx by Aaron Deich')

	for plot_index in range(4):
		x1, y1 = create_graphing_data(four_x_values_list[plot_index])
		plt.subplot(2, 2, plot_index + 1)
		plt.plot(x1, y1)
		plt.title('$x = {}$'.format(four_x_values_list[plot_index]))
		plt.xlabel('N'); plt.ylabel('Error')

	plt.savefig('HW3.png')
	plt.show()


def main():
	plot_2by2_grid([10., 2., -2., -10.])


main()
