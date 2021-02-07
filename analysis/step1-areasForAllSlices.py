import math
import numpy as np
import matplotlib.pyplot as plt

def ReadFile(path, fileName):
	print("Data File Path : {}".format(path))
	print("File Name : {}".format(fileName))

	# read
	f = open(path+fileName)
	lines = f.readlines()

	Name	= []
	Area	= []

	for line in lines:
		line = line.strip().split()

		name = line[0]
		Name.append(name)

		Area.append(float(line[1]))
	
	return Name, Area

if __name__=='__main__':
	print('hello')

	path = '../step1-PCA/step2-volumeEstimation/build-volumeEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'data_areaOfEachSlice.txt'
	Name, Area = ReadFile(path, fileName)

	SliceID = [int(name.split('-')[0]) for name in Name]
	print('SliceID: ', SliceID)

	# ground truth
	radius = 0.125 # unit : meter
	area_GT = math.pi * (radius**2)

	# error
	error = [abs(area-area_GT) for area in Area]
	error = np.array(error)
	print('error',error)

	# draw plot
	fig = plt.figure(dpi=128,figsize=(6,7))

	#
	# figure 1
	#
	ax = fig.add_subplot(211)
	ax.errorbar(SliceID, Area, yerr=error, marker='s')

	# set lable 
	ax.set_xlabel('SliceID', fontsize=10)
	ax.set_ylabel(r'Area / $m^2$', fontsize=10)

	# set limits
	#ax.set_xlim(-500, 500)
	#ax.set_ylim(-500, 500)

	# set title
	ax.set_title('Areas of all slices', fontsize=15)

	#
	# figure 2
	#
	ax = fig.add_subplot(212)

	error_percent = error*100.
	ax.plot(SliceID, error_percent, marker='s')

	# set lable 
	ax.set_xlabel('SliceID', fontsize=10)
	ax.set_ylabel('Accuracy / %', fontsize=10)

	# set limits
	#ax.set_xlim(-500, 500)
	#ax.set_ylim(-500, 500)

	# set title
	ax.set_title('Accuracies of the areas', fontsize=15)

	# overlapping 
	plt.tight_layout()

	# legend
	#plt.legend()

	# save figure
	plt.savefig('figure_step1_areaAndAccuracies.png')

	# print figure on screen
	plt.show()
