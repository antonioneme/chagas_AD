from skimage import io
import	random
import	sys
import	argparse
import	seaborn as sns
import	matplotlib.pyplot as plt
import	numpy as np
import	pandas as pd
from scipy import stats, integrate
from skimage.color import rgb2gray

import skimage.feature as feature

from scipy.spatial import distance

import cv2


#import statsmodels.api as sm



def	possible_values(Cl, X, Y):
	L = []
	for i in range(X):
		for j in range(Y):
			v = Cl[i][j]
			print ("v = ", v)
			if v not in L:
				L.append(v)
	return L

# Extract the k closest points to (x,y) from newLung
# Short cut: extract the poitns in the square delimited by (x-k), (y-k) - (x+k),y+k)
def	extract_neighs(newLung, x, y, k, Xo, Yo, Gray):
	R = []
	G = []
	B = []
	mndist = 100000.0
	mxdist = -100000.0
	canvas_dist_mn = 10000.0
	canvas_dist_mx = -10000.0
	V1 = np.array(newLung[x][y])
	C1 = np.array([x,y])
	sx = 0
	sy = 0

	windowR = [None] * (2*k+1)
	for ii in range(2*k+1):
		windowR[ii] = [0] * (2*k+1)

	windowG = [None] * (2*k+1)
	for ii in range(2*k+1):
		windowG[ii] = [0] * (2*k+1)

	windowB = [None] * (2*k+1)
	for ii in range(2*k+1):
		windowB[ii] = [0] * (2*k+1)

	windowGray = [None] * (2*k+1)
	for ii in range(2*k+1):
		windowGray[ii] = [0] * (2*k+1)

	for i in range(x - k, x + k + 1):
		for j in range(y - k, y + k + 1):
			if i > 0 and i < Xo and j > 0 and j < Yo:
				R.append(newLung[i][j][0])
				G.append(newLung[i][j][1])
				B.append(newLung[i][j][2])
				if i != x and j != y:
					V2 = np.array(newLung[i][j])
					ds = distance.cdist([V1], [V2], 'euclidean')[0][0]
					#print ("V = ", V1, V2, ds)
					#cc = sys.stdin.read(1)
					if ds > mxdist:
						mxdist = ds
						C2 = np.array([i,j])
						canvas_dist_mx = distance.cdist([C1], [C2], 'euclidean')[0][0]
					if ds < mndist:
						mndist = ds
						C2 = np.array([i,j])
						canvas_dist_mn = distance.cdist([C1], [C2], 'euclidean')[0][0]
				#NeighList.append(newLung[i][j])
				windowR[sx][sy] = newLung[i][j][0]
				windowG[sx][sy] = newLung[i][j][1]
				windowB[sx][sy] = newLung[i][j][2]

				windowGray[sx][sy] = Gray[i][j]
			else:

				windowR[sx][sy] = 0
				windowG[sx][sy] = 0
				windowB[sx][sy] = 0
				windowGray[sx][sy] = 0
				R.append(0)
				G.append(0)
				B.append(0)

			sy = sy + 1

		sx = sx + 1
		sy = 0

	windowR = np.array(windowR)
	windowG = np.array(windowG)
	windowB = np.array(windowB)
	windowGray = np.array(windowGray)
	#print ("Gray = ", i, j, x, y, windowGray)
	#print ("R = ", i, j, x, y, windowR)

	# https://medium.com/mlearning-ai/color-shape-and-texture-feature-extraction-using-opencv-cb1feb2dbd73

	gR = feature.greycomatrix(windowR, [1], [0, np.pi/2], levels=256, normed=True, symmetric=True)
	contrastR = feature.greycoprops(gR, 'contrast')
	dissimilarityR = feature.greycoprops(gR, 'dissimilarity')
	homogeneityR = feature.greycoprops(gR, 'homogeneity')
	energyR = feature.greycoprops(gR, 'energy')
	correlationR = feature.greycoprops(gR, 'correlation')

	gG = feature.greycomatrix(windowG, [1], [0, np.pi/2], levels=256, normed=True, symmetric=True)
	contrastG = feature.greycoprops(gG, 'contrast')
	dissimilarityG = feature.greycoprops(gG, 'dissimilarity')
	homogeneityG = feature.greycoprops(gG, 'homogeneity')
	energyG = feature.greycoprops(gG, 'energy')
	correlationG = feature.greycoprops(gG, 'correlation')

	gB = feature.greycomatrix(windowB, [1], [0, np.pi/2], levels=256, normed=True, symmetric=True)
	contrastB = feature.greycoprops(gB, 'contrast')
	dissimilarityB = feature.greycoprops(gB, 'dissimilarity')
	homogeneityB = feature.greycoprops(gB, 'homogeneity')
	energyB = feature.greycoprops(gB, 'energy')
	correlationB = feature.greycoprops(gB, 'correlation')

	gGray = feature.greycomatrix(windowGray, [1], [0, np.pi/2], levels=256, normed=True, symmetric=True)
	contrastGray = feature.greycoprops(gGray, 'contrast')
	dissimilarityGray = feature.greycoprops(gGray, 'dissimilarity')
	homogeneityGray = feature.greycoprops(gGray, 'homogeneity')
	energyGray = feature.greycoprops(gGray, 'energy')
	correlationGray = feature.greycoprops(gGray, 'correlation')

	#print ("texture = ", contrast, dissimilarity, homogeneity, energy, correlation)
	#print ("TX = ", contrast[0][1], dissimilarity[0][1], homogeneity[0][1], energy[0][1], correlation[0][1])
	#cc = sys.stdin.read(1)

	avg = [np.mean(R), np.mean(G), np.mean(B)]
	mn = [min(R), min(G), min(B)]
	mx = [max(R), max(G), max(B)]

	return [ [R, G, B], avg, mn, mx, [mndist, mxdist], [canvas_dist_mn, canvas_dist_mx], [contrastR[0][0], contrastR[0][1]], [dissimilarityR[0][0], dissimilarityR[0][1]], [homogeneityR[0][0], homogeneityR[0][1]], [energyR[0][0], energyR[0][1]], [correlationR[0][0], correlationR[0][1]],     [contrastG[0][0], contrastG[0][1]], [dissimilarityG[0][0], dissimilarityG[0][1]], [homogeneityG[0][0], homogeneityG[0][1]], [energyG[0][0], energyG[0][1]], [correlationG[0][0], correlationG[0][1]],     [contrastB[0][0], contrastB[0][1]], [dissimilarityB[0][0], dissimilarityB[0][1]], [homogeneityB[0][0], homogeneityB[0][1]], [energyB[0][0], energyB[0][1]], [correlationB[0][0], correlationB[0][1]],     [contrastGray[0][0], contrastGray[0][1]], [dissimilarityGray[0][0], dissimilarityGray[0][1]], [homogeneityGray[0][0], homogeneityGray[0][1]], [energyGray[0][0], energyGray[0][1]], [correlationGray[0][0], correlationGray[0][1]]       ]

def	normalize(atts):
	nAtts = []
	mn = min(atts)
	mx = max(atts)
	r = mx - mn
	for v in atts:
		nv = (v - mn) / r
		nAtts.append(nv)

	return nAtts
	

def	save_atts(FF, Pts, attsLung):
	# attLung[p] = [ [HR, HG, HB], avgRGB, mnRGB, mxRGB, distRGB, distCanvas ]

	f = open(FF, "w")
	for p in Pts:
		for elm in attsLung[p]:
			for vv in elm:
				f.write(str(vv) + "\t")
		f.write( str(p[0]) + "_" + str(p[1]) + "\n")
	f.close()

def	save_csv(FF, X, Y, Lung):
	f = open(FF, "w")
	for i in range(X):
		for j in range(Y):
			for kk in range(3):
				for elm in Lung[i][j][kk]:
					f.write(str(i) + "\t" + str(j) + "\t" + str(elm) + "\t")
			for kk in range(3, len(Lung[i][j])):
				f.write(str(i) + "\t" + str(j) + "\t" + str(elm) + "\t")
	f.close()
	

"""
This program reads a lung triplet consisting of:
1. The original image
2. The mask for the lung
3. The <<class>> of the pixel

python extract_atts_lung.py  -i Train/tr_im8.png  -m  LungMask/tr_lungmask8.png  -c Mask/tr_mask8.png  -o results/im_8.csv  -res_images/im_8.png
"""
parser = argparse.ArgumentParser()
parser.add_argument('-i', action = "store", dest = "i", help = "The input file containing the original lung image")
#parser.add_argument('-m', action = "store", dest = "m", help = "The input file containing the lung mask (what is lung)")
#parser.add_argument('-c', action = "store", dest = "c", help = "The input file containing the class of the lung  ground-glass (1), consolidation(2), pleural effusion (3)")
parser.add_argument('-r', action = "store", dest = "r", help = "The output file containing the data matrix,, normalized for each pixel ")
parser.add_argument('-rr', action = "store", dest = "rr", help = "The output file containing the data matrix,, normalized globally")
parser.add_argument('-o', action = "store", dest = "o", help = "The output file containing the image")
parser.add_argument('-s', action = "store", dest = "s", help = "The output file containing the image in csv")
parser.add_argument('-o2', action = "store", dest = "o2", help = "The output file containing the image when averaged by nk neighbors")
parser.add_argument('-nk', action = "store", dest = "nk", help = "The number of neighbos of each pixel to compare with")

args = parser.parse_args()

"""
Origx  = io.imread(args.i)
Orig = rgb2gray(Origx)
"""
Orig  = io.imread(args.i)

print ("s = ", Orig.shape)
Xo = Orig.shape[0]
Yo = Orig.shape[1]
#[Xo, Yo] =  Orig.shape
print ("S = ", Xo, Yo)

#cc = sys.stdin.read(1)

#Values_Orig = possible_values(Orig, Xo, Yo)
#print "Orig = ", Orig[49]
#print "PV Orig = ", Values_Orig

#Lung  = io.imread(args.m)
# Values: 0, 128
#print "Lung = ", Lung[49], Lung[380]
#[Xl, Yl] =  Lung.shape
#print ("S = ", Xl, Yl)
#Values_Lung = possible_values(Lung, Xl, Yl)
#print ("PV Lung = ", Values_Lung)
# PV Lung =  [0, 128, 255]

#Cl = io.imread(args.c)
# Values: 0, 128, 255
#[Xc, Yc] =  Cl.shape
#print "Cl = ", Cl[49], Cl[380]
#print ("S = ", Xc, Yc)
#Values_Cl = possible_values(Cl, Xc, Yc)
#print "PV CL = ", Values_Cl
# PV CL =  [0, 170, 85, 255]
# PV CL =  [0, 255, 128]
# PV CL =  [0, 255]
# Inconsistencies...?

newLung = Orig.copy()
for i in range(Xo):
	for j in range(Yo):
		"""
		if Lung[i][j] > 0:
			newLung[i][j] = Orig[i][j]
		else:
			newLung[i][j] = 0
		"""
		newLung[i][j] = Orig[i][j]


#newLung = apply_mask(Orig, Xo, Yo, Lung, Xl, Yl)
#print "nL = ", newLung[5]
io.imsave(args.o, newLung)

# Create the feature vector. For each vector filtered in (newLung), select 
# some features

newLungNeigh = newLung.copy()

grayLung = cv2.cvtColor(newLung, cv2.COLOR_BGR2GRAY)

nk = int(args.nk)
attsLung = {}
attsLung_nonorm = {}
for i in range(Xo):
	if i % 50 == 0:
		print ("row = ", i)
	for j in range(Yo):
		#print ("j = ", j, newLung[i][j])
		if 1 == 1:
		#if newLung[i][j] > 0:
			#the value of its k nearest neighbors
			# What parameters are to be extracted from the neoghborhood
			# of pixel (i,j)?
			[ RGB, avgRGB, mnRGB, mxRGB, distRGB, distCanvas, contrastR, dissimilarityR, homogeneityR, energyR, correlationR,  contrastG, dissimilarityG, homogeneityG, energyG, correlationG,  contrastB, dissimilarityB, homogeneityB, energyB, correlationB,   contrastGray, dissimilarityGray, homogeneityGray, energyGray, correlationGray ] = extract_neighs(newLung, i, j, nk, Xo, Yo, grayLung)
			# [ [R, G, B], avg, mn, mx, [mndist, mxdist], [canvas_dist_mn, canvas_dist_mx] ]
			#print ("X = ", i, j, contrast, dissimilarity, homogeneity, energy, correlation)
			#cc = sys.stdin.read(1)

			HR = stats.entropy(RGB[0])
			HG = stats.entropy(RGB[1])
			HB = stats.entropy(RGB[2])

			atts = [ [HR, HG, HB], avgRGB, mnRGB, mxRGB, distRGB, distCanvas, contrastR, dissimilarityR, homogeneityR, energyR, correlationR,  contrastG, dissimilarityG, homogeneityG, energyG, correlationG,  contrastB, dissimilarityB, homogeneityB, energyB, correlationB,  contrastGray, dissimilarityGray, homogeneityGray, energyGray, correlationGray ]
			#attsN = normalize(atts)
			attsN = atts
			#attsLung[(i,j)] = atts
			attsLung[(i,j)] = attsN
			# The non-normilized attributes
			attsLung_nonorm[(i,j)] = atts

			#print ("v = ", i, j, atts)
			#cc = sys.stdin.read(1)

			#newLungNeigh[i][j] = avg
		#else:
			#newLungNeigh[i][j] = 0

# Normalize data considering all vectors for each attribute
kK = attsLung.keys()
L = []
for k in kK:
	L.append(attsLung_nonorm[k])

L = np.array(L)

"""w
ln = len(attsLung[k])
mn = [1000000.0] * ln
mx = [-1000000.0] * ln
R = [0.0] * ln
for i in range(ln):
	#print "L0 = ", L[0], len(L[0])
	#print "i = ", i
	P = L[:,i]
	#print "P = ", P
	mn[i] = min(P)
	mx[i] = max(P)
	R[i] = mx[i] - mn[i]

attsLungN = {}
for k in kK:
	L = []
	for i,a in enumerate(attsLung_nonorm[k]):
		v = (a - mn[i]) / R[i]
		L.append(v)
	attsLungN[k] = L
Pts = attsLungN.keys()
save_atts(args.rr, Pts, attsLungN)
"""

Pts = attsLung.keys()
save_atts(args.r, Pts, attsLung)

#io.imsave(args.o2, newLungNeigh)

#save_csv(args.s, Xo, Yo, newLungNeigh)
print ("done")
