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


	#
	# step 1 : draw 3D scatters of the human 
	#
	path = '../step1-PCA/step2-volumeEstimation/build-volumeEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'data_slicesHumanModel.txt'
	ShapeList, ShapeName, X, Y, Z = ReadFile(path, fileName)

	print('ShapeList: ', ShapeList)

	# figure
	fig = plt.figure(dpi=128,figsize=(8,8))
	ax = fig.add_subplot(111, projection='3d')

	for i, name in enumerate(ShapeList):
		#print('shapeName:',name)

		shapeID = int(name.split('-')[0])
		print('shapeID', shapeID, 'shapeID/10',math.ceil(shapeID/10))
		marker = 'o'
		if(math.ceil(shapeID/10)==0):
			marker = 'o'
		elif(math.ceil(shapeID/10)==1):
			marker = 's'

		XCur = []
		YCur = []
		ZCur = []
		for j, nameCur in enumerate(ShapeName):
			if name==nameCur:
				XCur.append(X[j])
				YCur.append(Y[j])
				ZCur.append(Z[j])

		XCur = np.array(XCur)
		YCur = np.array(YCur)
		ZCur = np.array(ZCur)

		# Random sampling
		size = int(len(XCur)/1)
		index = np.random.choice(XCur.shape[0], size, replace=False)
		XCur = XCur[index]
		YCur = YCur[index]
		ZCur = ZCur[index]

		# draw
		ax.scatter(XCur, YCur, ZCur, s=10, cmap="jet", marker=marker, label=name)

	# set lable 
	ax.set_xlabel('X', fontsize=10)
	ax.set_ylabel('Y', fontsize=10)
	ax.set_zlabel('Z', fontsize=10)

	# set limits
	ax.set_xlim(-200, 200)
	ax.set_ylim(-200, 200)
	ax.set_zlim(-200, 200)

	# set title
	plt.title('Slices of Normalized Human', fontsize=15)
	
	# overlapping 
	plt.tight_layout()

	# legend
	plt.legend()

	# save figure
	plt.savefig('figure_step1_slicesHumanModel.png')

	# print figure on screen
	plt.show()

