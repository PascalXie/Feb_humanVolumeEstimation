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

	counter = 0
	for line in lines:
		line = line.strip().split()

		x		= float(line[0])
		y 	  	= float(line[1])
		z	  	= float(line[2])
		flag	= float(line[6])

		if flag==0:
			continue

		X.append(x)
		Y.append(y)
		Z.append(z)
	
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)

	return X, Y, Z

if __name__=='__main__':
	print('hello')

	# figure
	fig = plt.figure(dpi=128,figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')

	#
	# step 1 : draw 3D scatters of the original point cloud
	#
	path = '../step1-PCA/step1-pointCloud/build-3DCameraNormalEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'data_NorMap_RGBCam_Control_350000.txt'
	X, Y, Z = ReadFile(path, fileName)

	# Random sampling
	size = int(len(X)/50)
	index = np.random.choice(X.shape[0], size, replace=False)
	XCur = X[index]
	YCur = Y[index]
	ZCur = Z[index]

	# draw
	ax.scatter(XCur, YCur, ZCur, s=1, cmap="jet", marker='o')

	#
	# step 3 : settings of the plot
	#
	# set lable 
	ax.set_xlabel('X', fontsize=10)
	ax.set_ylabel('Y', fontsize=10)
	ax.set_zlabel('Z', fontsize=10)

	# set limits
	#ax.set_xlim(-400, 400)
	#ax.set_ylim(-400, 400)
	#ax.set_zlim(0, 800)

	# set title
	plt.title('Original Point Cloud', fontsize=15)
	
	# overlapping 
	plt.tight_layout()

	# legend
	plt.legend()

	# save figure
	plt.savefig('figure_step1_originalPointCloud.png')

	# print figure on screen
	plt.show()

