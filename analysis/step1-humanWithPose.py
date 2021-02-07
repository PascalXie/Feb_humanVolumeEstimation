import math
import numpy as np
import matplotlib.pyplot as plt

def ReadFile(path, fileName):
	print("Data File Path : {}".format(path))
	print("File Name : {}".format(fileName))

	# read
	f = open(path+fileName)
	lines = f.readlines()

	ShapeList = []
	ShapeName = []
	X		= []
	Y		= []
	Z		= []

	counter = 0
	for line in lines:
		line = line.strip().split()

		name	= str(line[0])
		x		= float(line[1])
		y 	  	= float(line[2])
		z	  	= float(line[3])

		if name not in ShapeList:
			ShapeList.append(name)

		ShapeName.append(name)
		X.append(x)
		Y.append(y)
		Z.append(z)
	
	X = np.array(X)
	Y = np.array(Y)
	Z = np.array(Z)

	return ShapeList, ShapeName, X, Y, Z


if __name__=='__main__':
	print('hello')

	# figure
	fig = plt.figure(dpi=128,figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')

	#
	# step 1 : draw 3D scatters of the human 
	#
	path = '../step1-PCA/step2-volumeEstimation/build-volumeEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'data_human.txt'
	ShapeList, ShapeName, X, Y, Z = ReadFile(path, fileName)

	# Random sampling
	size = int(len(X)/10)
	index = np.random.choice(X.shape[0], size, replace=False)
	XCur = X[index]
	YCur = Y[index]
	ZCur = Z[index]

	# draw
	ax.scatter(XCur, YCur, ZCur, s=10, cmap="jet", marker='o', label=ShapeList[0])

	# for step 2, find the size of the target along X, Y and Z axis
	lengthX = np.max(X) - np.min(X)
	lengthY = np.max(Y) - np.min(Y)
	lengthZ = np.max(Z) - np.min(Z)
	length = (lengthX**2+lengthY**2+lengthZ**2)**0.5
	print('length: ',length)

	#
	# step 2 : draw 3D scatters of the Canonical Coordinate System 
	#
	path = '../step1-PCA/step2-volumeEstimation/build-volumeEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'data_humanPose.txt'
	ShapeList, ShapeName, X, Y, Z = ReadFile(path, fileName)

	#print('CCS ShapeList size:',len(ShapeList),'; ShapeList: ',ShapeList)

	centor = np.array([X[0],Y[0],Z[0]])

	for i in range(3):
		vector = np.array([X[i+1],Y[i+1],Z[i+1]])
		lengthV = (vector[0]**2+vector[1]**2+vector[2]**2)**0.5
		print('lengthV: ',lengthV)
		vector = vector * length/lengthV * 0.5 # half the length of the target, due to the axis starts at the origin of the target
		axis = np.linspace(centor, centor+vector, 30)
	
		# draw
		ax.scatter(axis[...,0], 
				   axis[...,1], 
				   axis[...,2], 
				   s=10, cmap="jet", marker='s', label='Axis_'+str(i))

	#
	# step 3 : draw 3D scatters of the human 
	#
	fileName = 'data_humanPoseNormalized.txt'
	ShapeList, ShapeName, X, Y, Z = ReadFile(path, fileName)

	# Random sampling
	size = int(len(X)/10)
	index = np.random.choice(X.shape[0], size, replace=False)
	XCur = X[index]
	YCur = Y[index]
	ZCur = Z[index]

	# draw
	ax.scatter(XCur, YCur, ZCur, s=10, cmap="jet", marker='o', label=ShapeList[0])

	#
	# step 4 : settings of the plot
	#
	# set lable 
	ax.set_xlabel('X', fontsize=10)
	ax.set_ylabel('Y', fontsize=10)
	ax.set_zlabel('Z', fontsize=10)

	# set limits
	ax.set_xlim(-500, 500)
	ax.set_ylim(-500, 500)
	ax.set_zlim(0, 1000)

	# set title
	plt.title('Detected Human', fontsize=15)
	
	# overlapping 
	plt.tight_layout()

	# legend
	plt.legend()

	# save figure
	plt.savefig('figure_step1_detectedHuman.png')

	# print figure on screen
	plt.show()

