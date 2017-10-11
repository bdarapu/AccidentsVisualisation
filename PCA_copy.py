import csv
import random
import settings
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from numpy import linalg as LA
import statistics
import operator
import numpy as np

#read the data into a dict column wise
def readDataColumns(filename):
	dict_dataColumns=[]
	with open(filename,"r") as fd:
		reader=csv.reader(fd)
		headers=next(reader)
		dict1={}
		for i in range(len(headers)):
			dict1[headers[i]]=[]
		for row in reader:
			for i in range(len(row)):
				dict1[headers[i]].append(int(row[i]))
	return dict1


# Read data from CSV to list of lists(each sample is a list)
def readData(filename):
	with open(filename,"r") as fd:
		reader=csv.reader(fd)
		headers=next(reader)
		list_data=list(reader)
	return list_data,headers

#COnvert string literal to int in lists
def StringtoInt(list_data):
	List_Data=[]
	for l in list_data:
		Li=[]
		for i in range(len(l)):
			Li.append(int(l[i]))
		List_Data.append(Li)
	return List_Data


#Random Sampling 
def RandomSampling(list_data):
	rand_smpl=[list_data[i] for i in random.sample(range(len(list_data)),int(settings.samplePercent*len(list_data)))]
	return rand_smpl
	

def kmeansModel(list_data):
	kmeans=KMeans(n_clusters=2 , random_state=0).fit(list_data)
	return kmeans


#Binning the data into lists as per kmeans clustering
def MakeClusters(list_data,n_clusters):
	ClustersDict={}
	kmeans=KMeans(n_clusters=n_clusters , random_state=0).fit(list_data)
	for i in range(n_clusters):
		ClustersDict[i]=[]
	labels=kmeans.labels_
	for i in range(len(labels)):
		ClustersDict[labels[i]].append(list_data[i])
	return ClustersDict,kmeans.cluster_centers_


#Compute sum of squared errors for clusterDict
def GetSSE(ClustersDict,ClusterCentres):
	SSE=0
	for i in range(len(ClustersDict)):
		dist=0
		for j in range(len(ClustersDict[i])):
			dist+=distance.euclidean(ClustersDict[i][j],ClusterCentres[i])
		SSE+=dist
	return SSE

#plot elbow graph
def PlotElbow(SSE):
	X=[]
	Y=[]
	for i in range(1,len(SSE)):
		X.append(i)
		Y.append(SSE[i])
	plt.plot(X,Y,"-o")
	plt.show()

# Stratified Sampling
def StratifiedSample(ClustersDict):
	StratSample=[]
	for i in range(len(ClustersDict)):
		StratSample.extend(RandomSampling(ClustersDict[i]))
	return StratSample

#compute variance column wise and return sorted variances in descending order
def ComputeVariance(dict_dataColumns):
	dataVariance={}
	columnKeys=dict_dataColumns.keys()
	for k in (columnKeys):
		dataVariance[k]=0
	for k in (columnKeys):
		dataVariance[k]=statistics.variance(dict_dataColumns[k])
	sorted_dataVariance=sorted(dataVariance.items() , key=operator.itemgetter(1) , reverse=True)
	return sorted_dataVariance

def WriteToCsv(data,filename):
	with open(filename+'.csv',"w") as fd:
		wr=csv.writer(fd)
		wr.writerows(data)


#Task2 on sampled data from task1
def DimensionReduction(filename):
	dict_dataColumns=readDataColumns(filename)
	list_dataColumns=[]
	list_columnorder=[]
	dict_dataCovariance={}
	for k in dict_dataColumns.keys():
		list_columnorder.append(k)
		list_dataColumns.append(dict_dataColumns[k])

	for k in list_columnorder:
		dict_dataCovariance[k]=[]
	
	list_dataCovariance=np.cov(list_dataColumns)
	# print(list_dataCovariance)
	
	# list_dataCorelation=np.corrcoef(list_dataColumns)
	# print(list_dataCorelation)

	for i,k in enumerate(list_columnorder):
		dict_dataCovariance[k]=list_dataCovariance[i]

	
	eigenValues=LA.eigvals(list_dataCovariance)
	# eigenValuesCorr=LA.eigvals(list_dataCorelation)

	screePlot=[]
	for i,ev in enumerate(eigenValues):
		li=[i,ev]
		screePlot.append(li)



	# X=[]
	# for i in range(len(eigenValues)):
	# 	X.append(i)

	# plt.plot(X,eigenValues)
	# plt.show()

	# calculating eigen values with correlation
	# X=[]
	# for i in range(len(eigenValuesCorr)):
	# 	X.append(i)

	# plt.plot(X,eigenValuesCorr)
	# plt.show()

	dataVariance=ComputeVariance(dict_dataColumns)

	# print("Three attributes with highest PCA loadings for ",filename,"are: " ,dataVariance[:3])

	return screePlot
	


# def ComputeCoVariance(dict_dataColumns):
# 	dataCovariance={}
# 	for k in dict_dataColumns.keys():
# 		dataCovariance[k]=[]
# 	for k in dict_dataColumns.keys():



if __name__=='__main__':
	
	list_data,headers=readData(settings.filename)

	list_data=StringtoInt(list_data)
	
	randSampleData=RandomSampling(list_data)
	randSampleData.insert(0,headers)
	WriteToCsv(randSampleData,'RandomSamplingData')
	
	SSE=[]
	SSE.append(0)
	for i in range(1,settings.maxK):
		ClustersDict,ClusterCentres=MakeClusters(list_data,i)
		SSE.append(GetSSE(ClustersDict,ClusterCentres))
	
	# PlotElbow(SSE)
	optK=3

	ClustersDict,ClusterCentres=MakeClusters(list_data,optK)

	StratSampleData=StratifiedSample(ClustersDict)
	StratSampleData.insert(0,headers)
	WriteToCsv(StratSampleData,'StratifiedSamplingData')

	screePlotRand=DimensionReduction(settings.RandFilename)

	# print(screePlotRand)

	screePlotStrat=DimensionReduction(settings.StratFilename)

	# print(screePlotStrat)

	
	





	# print(sum(eigenValues))
	
	# 
	# print(dataVariance)

	# sum_dVar=0
	# for dVar in dataVariance:
	# 	sum_dVar+=dVar[1]
	# print(sum_dVar)

	# print(len(dataVariance))

	# pca = PCA(n_components=settings.intrinsicDimensionality,svd_solver='full')

	# pca.fit(list_data)

	# pca_values=pca.explained_variance_ratio_
	# print("explained variance: ",pca.explained_variance_)
	# print(pca_values)
	# X=[]
	# for i in range(settings.intrinsicDimensionality):
	# 	X.append(i)
	# plt.plot(X,pca_values)
	# plt.show()
	# print(pca.explained_variance_ratio_) 

	# print(LA.eig(list_data))

	# print(len(StratSample))

		# print(len(ClustersDict[i-1]))
		# print(ClusterCentres)

	# kmeans=kmeansModel(list_data)
	# # print(kmeans.cluster_centers_)
	# print(kmeans.labels_)
	# print(list_data[0])
	# print(len(list_data))
	# print(len(randSampleData))
