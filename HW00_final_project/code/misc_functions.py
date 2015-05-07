import numpy as np
import matplotlib.pyplot as plt



def quick_plot(x_and_y_arrays, title, figsize=(8, 6),
	show_plot=True, filename=None, ylog=False, xlog=False,
	xlabel=None, ylabel=None, xlim=None, ylim=None):

	fig = plt.figure(figsize=figsize, dpi=100, facecolor='w')
	ax = fig.add_subplot(111)
	for array_pair in x_and_y_arrays:
		ax.plot(array_pair[0], array_pair[1])
	plt.suptitle(title, fontsize=14)

	if xlim:
		ax.set_xlim(xlim)
	if ylim:
		ax.set_ylim(ylim)
	if xlabel:
		ax.set_xlabel(xlabel)
	if ylabel:
		ax.set_ylabel(ylabel)
	if ylog:
		ax.set_yscale('log')
	if xlog:
		ax.set_xscale('log')

	#ax.plot(traj2.t, traj2.x, label='$z_0 = 20.01$')
	#plt.xlabel('$t$ ', size=14);  plt.ylabel('$x(t)$ ', size=14);
	#ax.legend(loc='upper left')

	if filename:
		plt.savefig(filename)
	if show_plot:
		plt.show()




def test():
	x_array = np.linspace(0, 4*np.pi, 500)
	y_array = np.sin(x_array)
	quick_plot(x_array, y_array, title='sin(x)', filename='sincurve.png', xlog=True, 
		xlabel='$\sin (x)$', xlim=(1))
	


if __name__ == '__main__':
	test()

