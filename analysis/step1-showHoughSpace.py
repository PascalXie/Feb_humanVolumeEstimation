import math
import numpy as np
import matplotlib.pyplot as plt

def ReadFile(path, fileName):
	print("Data File Path : {}".format(path))
	print("File Name : {}".format(fileName))

	# read
	f = open(path+fileName)
	lines = f.readlines()

	X		= []
	Y		= []
	Z		= []
	Counts	= []

	for line in lines:
		line = line.strip().split()

		x		= float(line[0])
		y 	  	= float(line[1])
		z	  	= float(line[2])
		count  	= float(line[3])

		X.append(x)
		Y.append(y)
		Z.append(z)
		Counts.append(count)

	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)
	Counts = np.array(Counts)

	return X, Y, Z, Counts

if __name__=='__main__':
	print('hello')

	# figure
	fig = plt.figure(dpi=128,figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')

	#
	# part 2 : draw 3D scatters of the point clouds
	#

	path = '../step1-PCA/step2-volumeEstimation/build-volumeEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'data_histogram3d.txt'
	X, Y, Z, Counts = ReadFile(path, fileName)

	# generate histograms based on the counts
	CountMin = np.min(Counts)
	CountMax = np.max(Counts)
	SizeHistograms = 9 
	CountWidth = (CountMax-CountMin)/float(SizeHistograms)
	for i in range(SizeHistograms):

		#if i==0:
		#	continue

		countCurrent_min = CountMin + CountWidth*float(i)
		countCurrent_max = CountMin + CountWidth*float(i+1)
		print('\ncontur ID: ',i)
		print('countCurrent_min: ',countCurrent_min)
		print('countCurrent_max: ',countCurrent_max)

		XCur = []
		YCur = []
		ZCur = []

		for j in range(len(Counts)):
			countCurrent = Counts[j]
			if countCurrent>countCurrent_min and countCurrent<=countCurrent_max:
				XCur.append(X[j])
				YCur.append(Y[j])
				ZCur.append(Z[j])

		print('ZCur.size: ',len(ZCur))

		# draw
		sizeOfTheMarker = 5*(i+1)
		print('sizeOfTheMarker: ',sizeOfTheMarker)
		labelName = str(int(countCurrent_min/1000)) + '-' + str(int(countCurrent_max/1000)) + 'k'

		XCur = np.array(XCur)
		YCur = np.array(YCur)

		XCur = XCur/math.pi*180.
		YCur = YCur/math.pi*180.

		ax.scatter(XCur, YCur, ZCur, s=sizeOfTheMarker, cmap="jet", marker='o', label=labelName)

	
	# set lable 
	ax.set_xlabel(r'$\theta / degrees$', fontsize=10)
	ax.set_ylabel(r'$\phi / degrees$', fontsize=10)
	ax.set_zlabel('r / mm', fontsize=10)

	# set limits
	ax.set_xlim(0, 180)
	ax.set_ylim(0, 360)
	#ax.set_zlim(0, 800)

	# set title
	plt.title('3D Plane Detection with Hough Transformation', fontsize=15)
	
	# overlapping 
	plt.tight_layout()

	# legend
	plt.legend()

	# save figure
	plt.savefig('figure_step1_HoughSpace.png')


	# print figure on screen
	plt.show()
