# HW3.py Aaron Deich

# Comments on the plot behavior:
# Generally, when N is small, the Taylor approximation will be less precise.
# Thus all the graphs tend toward 0% fractional error as N gets large.
# When x is negative, x**i's error blows up with i, though N! eventually surpasses it.
# My best guess as to why part (b) improves the error is that when x is negative,
# S(-x) will have half negative terms (when i is even) and half positive (odd), and these will tend
# to cancel out, making a smaller sum number.

# Comments on my LaTeX setup:
# I'm running TeXShop version 3.23. It is working fine.

import numpy as np
import matplotlib.pyplot as plt

# Compute absolute fractional error (takes 2 numbers; returns 1 number).
def compute_abs_frac_error(true_number, approx_number):
	if true_number is None or approx_number is None:
		raise BaseException("got a None for x:{}, N:{}".format(true_number, approx_number))
	return np.abs((approx_number - true_number)/true_number)

# Creates taylor approximation array (int N) for a given float x.
def create_approximated_e_array(x, N, problem_part):
	# create empty array of length N.
	output_array = np.empty(N)

	# Force proper types.
	x = float(x); N = int(N)
	
	# auxiliary function which computes the ith term of the Taylor sum.
	def compute_sum_element_i(x, i):
		ith_term = (x**i)/np.math.factorial(i)
		return ith_term	

	def compute_whole_approx_array(x, N):
		output_array[0] = compute_sum_element_i(x, 0)
		for i in range(1, N):
			output_array[i] = output_array[i-1] + compute_sum_element_i(x, i)

		return output_array
	

	if problem_part == 'a' or x > 0:
		return compute_whole_approx_array(x, N)
	elif problem_part == 'b':
		return 1/compute_whole_approx_array(-x, N)
	else:
		raise BaseException('Wrong "problem_part" passed: you passed"{}"'.format(
			problem_part))

# Computes the dependent and independent axis arrays for
# the abs-frac-error for N=60 and specified x.  
def create_graphing_data(x, problem_part='a'):
	N = 60

	independent_axis = np.arange(N)

	# Using vectorization here.
	dependent_axis = compute_abs_frac_error(np.e**x,
		create_approximated_e_array(x, N, problem_part))

	return (independent_axis, dependent_axis)
	
# Create 2x2 grid containing 4 plots (one for each x-value).
def plot_2by2_grid(four_x_values_list):

	plt.figure(facecolor='w')
	plt.suptitle('Absolute Fractional Error of Nth-degree Taylor Sum $e^x$ by Aaron Deich')

	for plot_index in range(4):
		x1, ya = create_graphing_data(four_x_values_list[plot_index], problem_part='a')
		x1, yb = create_graphing_data(four_x_values_list[plot_index], problem_part='b')
		plt.subplot(2, 2, plot_index + 1)
		plt.plot(x1, ya, 'b-', label='part a')
		plt.plot(x1, yb, 'g--', label='part b')
		plt.title('$x = {}$'.format(four_x_values_list[plot_index]))
		plt.xlabel('N'); plt.ylabel('$|S(x,N)-e^x| / e^x$')
		plt.legend(loc='top right')
		plt.yscale('log')

	plt.tight_layout()
	plt.savefig('HW3.png')
	plt.show()


def main():
	plot_2by2_grid([10., 2., -2., -10.])


main()
