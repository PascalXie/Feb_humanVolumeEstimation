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
	X		= []
	Y		= []

	counter = 0
	for line in lines:
		line = line.strip().split()

		name = line[0]

		triX = []
		triY = []

		triX.append(float(line[1]))
		triY.append(float(line[2]))

		triX.append(float(line[3]))
		triY.append(float(line[4]))

		triX.append(float(line[5]))
		triY.append(float(line[6]))

		Name.append(name)
		X.append(triX)
		Y.append(triY)
	
	return Name, X, Y

def ReadFile_pixels(path, fileName):
	print("Data File Path : {}".format(path))
	print("File Name : {}".format(fileName))

	# read
	f = open(path+fileName)
	lines = f.readlines()

	Name	= []
	X		= []
	Y		= []

	counter = 0
	for line in lines:
		line = line.strip().split()

		name = line[0]
		Name.append(name)

		X.append(float(line[1]))
		Y.append(float(line[2]))
	
	return Name, X, Y


if __name__=='__main__':
	print('hello')

	path = '../step1-PCA/step2-volumeEstimation/build-volumeEstimation-Desktop_Qt_5_14_2_GCC_64bit-Debug/'
	fileName = 'data_DelaunayTriangles.txt'
	Name, X, Y = ReadFile(path, fileName)

	fileName = 'data_pixelsInsideTheSlices.txt'
	pixName, pixX, pixY = ReadFile_pixels(path, fileName)

	# get slice IDs
	SliceIDs = []
	for nameCur in Name:
		SliceID = int(nameCur.split('-')[0])
		if SliceID not in SliceIDs:
			SliceIDs.append(SliceID)

	print('SliceIDs: ',SliceIDs)
	print('SliceID: ',SliceIDs[-1])




	# figure
	fig = plt.figure(dpi=128,figsize=(8,8))

	# slices
	for i in range(4):
		ax = fig.add_subplot(221+i)
		SliceID = SliceIDs[-1*(i+1)]
		print('SliceID',SliceID)

		# draw triangles
		for j,name in enumerate(Name):
			triX = X[j]
			triY = Y[j]
			IDCurrent = int(name.split('-')[0])
			if IDCurrent==SliceID:
				#print('IDCurrent',IDCurrent)
				#print('triX',triX)
				CurTriX = [triX[0],triX[1],triX[2],triX[0]]
				CurTriY = [triY[0],triY[1],triY[2],triY[0]]
				ax.plot(CurTriX, CurTriY)

		# draw pixels 
		CurPixX = []
		CurPixY = []
		for j,name in enumerate(pixName):
			IDCurrent = int(name.split('-')[0])
			if IDCurrent==SliceID:
				#print('IDCurrent',IDCurrent)
				#print('triX',triX)
				CurPixX.append(pixX[j])
				CurPixY.append(pixY[j])

		ax.scatter(CurPixX, CurPixY, s=1, marker='s')

		# set lable 
		ax.set_xlabel('X', fontsize=10)
		ax.set_ylabel('Y', fontsize=10)

		# set limits
		ax.set_xlim(-500, 500)
		ax.set_ylim(-500, 500)

		# set title
		ax.set_title('Slice-'+str(SliceID), fontsize=15)
	
	# overlapping 
	plt.tight_layout()

	# legend
	#plt.legend()

	# save figure
	plt.savefig('figure_step1_HumanSlices.png')

	# print figure on screen
	plt.show()




		
