import numpy as np
import misc_functions as mf


# returns centered derivative of array a.
def take_derivative(a):
	n = len(a) - 2
	d = np.empty(n)
	for i in range(n):
		d[i] = (a[i+2] - a[i]) / 2.
	# pad with 1 zero on each end.
	padded_d = np.lib.pad(d, (1, 1), 'constant', constant_values = (0, 0))
	return padded_d

	
# I'm not sure why this is called fastsmooth. Basically this just
# calls sliding_avg() recursively. 
def fastsmooth(Y, w, smooth_iterations=None, ends=None):

	current_smooth_level_data = Y
	for nth_smooth in range(smooth_iterations):
		current_smooth_level_data = sliding_avg(current_smooth_level_data, w, ends)

	return current_smooth_level_data



# Smooths data by performing sliding average.
def sliding_avg(Y, smoothwidth, ends):
	width = np.round(smoothwidth)
	sum_points = sum(Y[0: width-1])
	s = np.zeros(len(Y))
	halfwidth = np.round(width / 2.)	
	for k in range(1, len(Y) - width):
		s[k + halfwidth -1] = sum_points
		sum_points = sum_points - Y[k]
		sum_points = sum_points + Y[k + width]	
	s[k + halfwidth] = np.sum(Y[len(Y) - width + 1: len(Y)])
	SmoothY = s / width

	# then taper the ends of the signal if ends=1.
	if ends == 1:
		startpoint = (smoothwidth + 1.) / 2.
		SmoothY[0] = (Y[0] + Y[2]) / 2.
		L = len(Y)
		for k in range(1, startpoint):
			SmoothY[k] = np.mean(Y[0: 2*k -1])
			SmoothY[L - k] = np.mean(Y[L - 2*k + 2: L])

	return SmoothY


# takes x-array, y-array, then several numbers.
def find_peaks(x, y, SlopeThreshold, AmpThreshold, smoothwidth, 
	peakgroup, smooth_iterations, polyfitdegree):

	poly_coefficients = np.polyfit(x, y, deg=polyfitdegree)
	polyfit_y_array = np.polyval(poly_coefficients, x)

	peakgroup = np.round(peakgroup)

	d = fastsmooth(take_derivative(y), smoothwidth, smooth_iterations)
	second_derivative = take_derivative(d)
		#mf.quick_plot([(x, fastsmooth(y, smoothwidth, smooth_iterations))], title='smoothed derivative',
		#	xlim=50, xlog=True, ylog=False)

	### for testing
	print 'len d: {}, len x: {}'.format(len(d), len(x))
#	padded_d = np.lib.pad(d, (1, 1), 'constant', constant_values = (0, 0))
	mf.quick_plot([(x, np.real(d))], title='derivative of FFT', xlog=True, xlim=(50.), filename='derivative.png')
	#second_padded_d = np.lib.pad(second_derivative, (2, 2), 'constant', constant_values = (0, 0))
	#mf.quick_plot(x, np.real(second_padded_d), title='second derivative', xlog=True, xlim=(50.))
	### for testing

	ZeroDerivatives = []

	starting_index = 2 * np.round(smoothwidth/2 - 1)
	ending_index =  len(y) - smoothwidth - 1


	for j in range(starting_index, ending_index):

		# Detects zero-crossing.
		if np.real(d[j]) * np.real(d[j+1]) < 0.:
			# If slope of derivative is larger than threshold.
			if np.abs((d[j] - d[j + 1])) > np.abs((SlopeThreshold * y[j])):
				pass
			# that is, if second-derivative is larger than threshhold.
			elif np.abs(second_derivative[j]) > np.abs(SlopeThreshold): 
				# If height of peak is larger than AmpThreshold.
				ZeroDerivatives.append({'x': x[j], 'y': y[j], 'j': j})
				if (np.abs(y[j]) - polyfit_y_array[j] > AmpThreshold):
					pass

	# For each zero-derivative, find local maximum signal value.
	# 'local' means within index search_range of guess index.
	actual_peak_values = []
	for zero_derivative_spot in ZeroDerivatives:
		xguess = zero_derivative_spot['x']
		yguess = zero_derivative_spot['y']
		jguess = zero_derivative_spot['j']
		search_range = 4 * smoothwidth 
		search_j_min = jguess - search_range; search_j_max = jguess + search_range + 1
		search_set = y[search_j_min : search_j_max]
		j_of_max = np.argmax(search_set) + search_j_min
		if y[j_of_max] - polyfit_y_array[j_of_max] > AmpThreshold:
			actual_peak_values.append({'x': x[j_of_max], 'y': y[j_of_max], 'j': j_of_max})
		else:
			pass

	simple_peak_xy_list = []
	for peak in actual_peak_values:
		simple_peak_xy_list.append([peak['x'], peak['y']])

	peaks_before_poly_subtraction = []

	# data set to pass through return statement. Also include 1st and 2nd deriv before smooth.
	all_data_pack = [polyfit_y_array, peaks_before_poly_subtraction, simple_peak_xy_list,
		d, second_derivative]

	# returns list of (x, y) tuples for each peak
	return np.array(simple_peak_xy_list)


def module_test():
	pass


