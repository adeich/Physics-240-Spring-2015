import numpy as np
import misc_functions as mf

# returns elements of x_array which are closest to value.
def indices_of_closest(x_array, value):
	difference_array = np.abs(x_array - value)
	indices_of_min = np.where(difference_array - np.min(difference_array) == 0)
	return indices_of_min, np.min(difference_array)
	

# returns centered derivative of a.
def take_derivative(a):
	n = len(a) - 2
	d = np.empty(n)
	for i in range(n):
		d[i] = (a[i+2] - a[i]) / 2.
	return d

	
# I'm not sure why this is called fastsmooth. Basically this just
# calls sliding_avg() recursively. 
def fastsmooth(Y, w, smooth_type=None, ends=None):
	if not smooth_type:
		smooth_type=1
		ends = 0	
	elif not ends:
		ends = 0

	if smooth_type == 1:
		SmoothY = sliding_avg(Y, w, ends)
	elif smooth_type == 2:
		SmoothY = sliding_avg(sliding_avg(Y, w, ends), w, ends)
	elif smooth_type == 3:
		SmoothY = sliding_avg(sliding_avg(sliding_avg(Y, w, ends), w, ends), w, ends)

	return SmoothY



# Smooths data by performing sliding average
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

def gaussfit(x, y):
	max_y = np.max(y)
	for p in range(len(y)):
		if y[p] < max_y/100.:
			y[p] = max_y/100.
	z = np.log(y)
	coef = np.polyfit(x, z, 2)
	a = coef[2];
	b = coef[1];
	c = coef[0];
	height = np.exp(a - c * (b/(2 * c))**2);
	position = -b / (2 * c);
	width= 2.35482 / (np.sqrt(2) * np.sqrt(-c));

	return height, position, width


def find_peaks(x, y, SlopeThreshold, AmpThreshold, smoothwidth, 
	peakgroup, smoothtype=None):


	# set smoothtype to a number between 1 and 3.
	if not smoothtype:
		smoothtype = 1 
	elif smoothtype > 3:
		smoothtype = 3
	elif smoothtype < 1:
		smoothtype = 1
	
	# set smoothwidth. used only in determining start- and end-indices. 
	if smoothwidth < 1:
		smoothwidth = 1
	smoothwidth = np.round(smoothwidth)
	
	peakgroup = np.round(peakgroup)

	if smoothwidth > 1:
	#	mf.quick_plot(x, fastsmooth(np.abs(y), smoothwidth, smoothtype), title='smoothed FFT',
	#		xlim=50, xlog=True, ylog=False)
		d = fastsmooth(take_derivative(y), smoothwidth, smoothtype)
	else:
		d = take_derivative(y)

	### for testing
	print 'len d: {}, len x: {}'.format(len(d), len(x))
	padded_d = np.lib.pad(d, (1, 1), 'constant', constant_values = (0, 0))
	mf.quick_plot(x, np.real(padded_d), title='derivative of FFT', xlog=True, xlim=(50.), filename='derivative.png')
	second_derivative = take_derivative(d)
	second_padded_d = np.lib.pad(second_derivative, (2, 2), 'constant', constant_values = (0, 0))
	mf.quick_plot(x, np.real(second_padded_d), title='second derivative', xlog=True, xlim=(50.))
	### for testing


	n = np.round((peakgroup/2.) + 1)
	P = []
	vectorlength = len(y)
	peak = 1.
	AmpTest = AmpThreshold
	SimplePeaks = []
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
			elif np.abs(second_padded_d[j]) > np.abs(SlopeThreshold): 
				ZeroDerivatives.append({'x': x[j], 'y': y[j], 'j': j})
				# If height of peak is larger than AmpThreshold.
				if (np.abs(y[j]) > AmpTest) or (np.abs(y[j + 1]) > AmpTest):
					SimplePeaks.append([np.abs(x[j]), np.abs(y[j])])

					xx = np.zeros(peakgroup)
					yy = np.zeros(peakgroup)

					for k in range(1, int(peakgroup)):
						groupindex = j + k - n + 1 
						if groupindex < 1: 
							groupindex = 1
						if groupindex > vectorlength:
							groupindex = vectorlength
						xx[k] = x[groupindex - 1]
						yy[k] = y[groupindex - 1]
					
					if peakgroup > 2:
						height, position, width = gaussfit(xx, yy)
						peakX = np.real(position)
						peakY = np.real(height)
						measured_width = np.real(width)
					else:
						peakY = np.max(yy)
						pindex = indices_of_closest(yy, peakY)
						peakX = xx[pindex[0]]
						measured_width = 0

					if peakY > AmpThreshold:
						P.append([np.round(peak), peakX, peakY, measured_width, 1.0646* peakY * measured_width])
						

	# For each zero-derivative, find local maximum signal value.
	# 'local' means within index search_range of guess index.
	actual_peak_values = []
	for zero_derivative_spot in ZeroDerivatives:
		xguess = zero_derivative_spot['x']
		yguess = zero_derivative_spot['y']
		jguess = zero_derivative_spot['j']
		search_range = 20
		search_j_min = jguess - search_range; search_j_max = jguess + search_range + 1
		search_set = y[search_j_min : search_j_max]
		j_of_max = np.argmax(search_set) + search_j_min
		actual_peak_values.append({'x': x[j_of_max], 'y': y[j_of_max], 'j': j_of_max})

	simple_peak_xy_list = []
	for peak in actual_peak_values:
		simple_peak_xy_list.append([peak['x'], peak['y']])

	# returns (x, first_derivative dydx), (x, second_derivative), list_of_peaks. 
	return np.array(simple_peak_xy_list)


