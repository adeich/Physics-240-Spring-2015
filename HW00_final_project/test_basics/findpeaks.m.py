import numpy as np

def find_peaks(x, y, SlopeThreshold, AmpThreshold, smoothwidth, 
	peakgroup, smoothtype=None):

	if not smoothtype:
		smoothtype = 1 
	elif smoothtype > 3:
		smoothtype = 3
	elif smoothtype < 1:
		smoothtype = 1
	
	if smoothwidth < 1:
		smoothwidth = 1
	smoothwidth = np.round(smoothwidth)
	
	peakgroup = np.round(peakgroup)

	if smoothwidth > 1:
		d = fastsmooth(deriv(y), smoothwidth, smoothtype)
	else:
		d = deriv(y)


	n = np.round((peakgroup/2) + 1)
	P = []
	vectorlength = len(y)
	peak = 1.
	AmpTest = AmpThreshold

	starting_index = 2 * np.round(smoothwidth/2 - 1)
	ending_index =  len(y) - smoothwidth - 1

	for j in range(starting_index, ending_index):

		# Detects zero-crossing.
		if np.sign(d[j]) < np.sign(d[j + 1]):
			# If slope of derivative is larger than threshold.
			if (d[j] - d[j + 1]) > (SlopeThreshold * y[j]):
				# If height of peak is larger than AmpThreshold.
				if ((y[j] > AmpTest) or (y[j + 1] > AmpTest):

					xx = np.zeros(len(peakgroup))
					yy = np.zeros(len(peakgroup))

					for k in range(1, peakgroup + 1):
						groupindex = j + k - n + 2
						if groupindex < 1: 
							groupindex = 1
						if groupindex > vectorlength:
							groupindex = vectorlength
						xx[k] = x[groupindex]
						yy[k] = y[groupindex]
					
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
						P.append(np.round(peak), PeakX, PeakY, MeasuredWidth, 1.0646.* PeakY * MeasuredWidth)
						
		


# returns elements of x_array which are closest to value.
def indices_of_closest(x_array, value):
	difference_array = np.abs(x_array - value)
	indices_of_min = np.where(difference_array - np.min(difference_array) == 0)
	return indices_of_min, np.min(difference_array)
	
def guassfit(x_array, y_array):
	pass

# returns centered derivative of a.
def deriv(a):
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


# Smooths data by performing sliding average
def sliding_avg(Y, smoothwidth, ends):
	width = np.round(smoothwidth)
	sum_points = sum(Y[0: w-1])
	s = np.zeros(len(Y))
	halfwidth = np.round(width / 2.)	
	for k in range(1, len(Y) - width):
		s[k + halfwidth -1] = sum_points
		sum_points = sum_points - Y[k]
		sum_points = sum_points + Y[k + width]	
	s[k + halfwidth] = np.sum(Y[len(Y) - width + 1: len(Y)])
	SmoothY = s / width
	# then taper the ends of the signal if ends=1.

# removes NaNs from data
def remove_NaN(a):
	pass

def fit_gaussian(x, y)
	maxy=max(y);
	for p=1:length(y),
   	if y(p)<(maxy/100),y(p)=maxy/100;end
	
	z = log(y)
	coef = np.polyfit(x,z,2)
	a=coef[3]
	b=coef[2]
	c=coef[1]
	Height= exp(a-c*(b/(2*c))^2);
	Position= -b / (2*c);
	Width=2.35482 / (sqrt(2)*sqrt(-c))

	return Height, Position, Width



