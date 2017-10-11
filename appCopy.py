from flask import Flask
from flask import render_template
import csv
import random
import settings
from sklearn.cluster import KMeans
from scipy.spatial import distance
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold import MDS
from numpy import linalg as LA
import statistics
import operator
import numpy as np
from sklearn import preprocessing

# app = Flask(__name__)

# @app.route("/")
# def index():
# 	return render_template("index.html")

# @app.context_processor
# def override_url_for():
#     return dict(url_for=dated_url_for)

# def dated_url_for(endpoint, **values):
#     if endpoint == 'static':
#         filename = values.get('filename', None)
#         if filename:
#             file_path = os.path.join(app.root_path,
#                                      endpoint, filename)
#             values['q'] = int(os.stat(file_path).st_mtime)
#     return url_for(endpoint, **values)

# @app.route("/data")
# def dataProcessing():
# 	#read the data into a dict column wise
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
				dict1[headers[i]].append(float(row[i]))
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
			Li.append(float(l[i]))
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
	# plt.plot(X,Y,"-o")
	# plt.show()
	return [X,Y]

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
	sorted_list = [list(elem) for elem in sorted_dataVariance]
	return sorted_list
	

	# return sorted_dataVariance

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
	screePlot.append(['i','ev'])
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

	

	# print("Three attributes with highest PCA loadings for ",filename,"are: " ,dataVariance[:3])

	return screePlot
	

def PCAdata(filename):
	dict_dataColumns=readDataColumns(filename)
	dataVariance=ComputeVariance(dict_dataColumns)
	top3_attributes=[]
	for i in range(len(dataVariance[:3])):
		top3_attributes.append(dataVariance[i][0])
	# print(top3_attributes)

	listTop3Col=[]
	for key in top3_attributes:
		listTop3Col.append(dict_dataColumns[key])
	# print(listTop3)
	listTop3=[]
	for i in range(len(listTop3Col[0])):
		li=[]
		for j in range(len(listTop3Col)):
			li.append(listTop3Col[j][i])
		listTop3.append(li)
	# print(len(listTop3))

	pca = PCA(n_components=2)
	pcaNumpy=pca.fit_transform(listTop3)
	pcaList=[]
	for i in range(len(pcaNumpy)):
		x=int(pcaNumpy[i][0])
		y=int(pcaNumpy[i][1])
		li=[x,y]
		pcaList.append(li)
	return pcaList

# def ComputeCoVariance(dict_dataColumns):
# 	dataCovariance={}
# 	for k in dict_dataColumns.keys():
# 		dataCovariance[k]=[]
# 	for k in dict_dataColumns.keys():

def MDSdata(filename,type_mds):
	dict_dataColumns=readDataColumns(filename)
	dataVariance=ComputeVariance(dict_dataColumns)
	top3_attributes=[]
	for i in range(len(dataVariance[:3])):
		top3_attributes.append(dataVariance[i][0])
	# print(top3_attributes)

	listTop3Col=[]
	for key in top3_attributes:
		listTop3Col.append(dict_dataColumns[key])
	# print(listTop3)
	listTop3=[]
	for i in range(len(listTop3Col[0])):
		li=[]
		for j in range(len(listTop3Col)):
			li.append(listTop3Col[j][i])
		listTop3.append(li)
	print(type_mds)
	dis_mat=pairwise_distances(listTop3,metric=type_mds)
	print(dis_mat)
	mds=MDS(n_components=2,dissimilarity='precomputed')
	mdsNumpy=mds.fit_transform(dis_mat)
	
	mdsList=[]
	for i in range(len(mdsNumpy)):
		x=float(mdsNumpy[i][0])
		y=float(mdsNumpy[i][1])
		li=[x,y]
		mdsList.append(li)
	return mdsList


def getTop3Attributes(filename):
	dict_dataColumns=readDataColumns(filename)
	dataVariance=ComputeVariance(dict_dataColumns)
	top3_attributes=[]
	for i in range(len(dataVariance[:3])):
		top3_attributes.append(dataVariance[i][0])
	# print(top3_attributes)

	listTop3Col=[]
	for key in top3_attributes:
		listTop3Col.append(dict_dataColumns[key])
	# print(listTop3)
	listTop3=[]
	for i in range(len(listTop3Col[0])):
		li=[]
		for j in range(len(listTop3Col)):
			li.append(listTop3Col[j][i])
		listTop3.append(li)
	listTop3.insert(0,top3_attributes)
	return listTop3

	# def MDSdata(pcaList):
# 	mds=MDS(n_components=2)
# 	mdsNumpy=mds.fit_transform(pcaList)
# 	mdsList=[]
# 	for i in range(len(mdsNumpy)):
# 		x=int(mdsNumpy[i][0])
# 		y=int(mdsNumpy[i][1])
# 		li=[x,y]
# 		mdsList.append(li)
# 	return mdsList


	
list_data,headers=readData(settings.filename)

# print(settings.filename)
# print(headers)

list_data=StringtoInt(list_data)
list_data = preprocessing.scale(list_data)

randSampleData=RandomSampling(list_data)
randSampleData.insert(0,headers)
WriteToCsv(randSampleData,'static/RandomSamplingData')
dict_dataColumns = readDataColumns(settings.RandFilename)
sorted_dataVariance = ComputeVariance(dict_dataColumns)
print(sorted_dataVariance)
# random_correlation_matrix =  np.corrcoef(randSampleData,rowvar=False)
# pca = PCA(n_components=3)
# random_pca_values = pca.fit_transform(random_correlation_matrix)
# random_scree_values = np.square(random_pca_values)
# random_scree_values = np.sum(random_scree_values, axis=1)
# print(random_scree_values)


# SSE=[]
# SSE.append(0)
# for i in range(1,settings.maxK):
# 	ClustersDict,ClusterCentres=MakeClusters(list_data,i)
# 	SSE.append(GetSSE(ClustersDict,ClusterCentres))

# # SSE=PlotElbow(SSE)
# SSEPlotData=[]
# for i,row in enumerate(SSE):
# 	SSEPlotData.append([i,SSE[i]])
# SSEPlotData.insert(1,['i','ev'])
# WriteToCsv(SSEPlotData[1:],'static/ElbowPlot')
# optK=3

# ClustersDict,ClusterCentres=MakeClusters(list_data,optK)

# StratSampleData=StratifiedSample(ClustersDict)
# StratSampleData.insert(0,headers)
# WriteToCsv(StratSampleData,'static/StratifiedSamplingData')

# screePlotRand=DimensionReduction(settings.RandFilename)
# WriteToCsv(screePlotRand,'static/screePlotRand')

# # print(screePlotRand)

# screePlotStrat=DimensionReduction(settings.StratFilename)
# WriteToCsv(screePlotStrat,'static/screePlotStrat')

# listScatterStrat=[]
# listScatterStrat=PCAdata(settings.StratFilename)
# # listScatterStrat=listScatterStrat.tolist()
# listScatterStrat.insert(0,['a1','a2'])
# # print(listScatterStrat[0])
# WriteToCsv(listScatterStrat,"static/ScatterPlotStrat")
# 	# print(screePlotStrat)

	
# listScatterRand=[]
# listScatterRand=PCAdata(settings.RandFilename)

# listScatterRand.insert(0,['a1','a2'])

# WriteToCsv(listScatterRand,"static/ScatterPlotRand")


# listScatterStratMds_euclidean=MDSdata(settings.StratFilename,"euclidean")
# listScatterStratMds_euclidean.insert(0,['a1','a2'])
# WriteToCsv(listScatterStratMds_euclidean,"static/StratMdsEuclidean")

# listScatterStratMds_correlation=MDSdata(settings.StratFilename,"correlation")
# listScatterStratMds_correlation.insert(0,['a1','a2'])
# WriteToCsv(listScatterStratMds_correlation,"static/StratMdscorrelation")

# listScatterRandMds_euclidean=MDSdata(settings.RandFilename,"euclidean")
# listScatterRandMds_euclidean.insert(0,['a1','a2'])
# WriteToCsv(listScatterRandMds_euclidean,"static/RandMdsEuclidean")

# listScatterRandMds_correlation=MDSdata(settings.RandFilename,"correlation")
# listScatterRandMds_correlation.insert(0,['a1','a2'])
# print(listScatterRandMds_correlation)
# WriteToCsv(listScatterRandMds_correlation,"static/RandMdsCorrelation")

# HighestPCAStrat=getTop3Attributes(settings.StratFilename)
# HighestPCAStrat.pop(0)
# HighestPCAStrat = preprocessing.scale(HighestPCAStrat)
# HighestPCAStrat = HighestPCAStrat.tolist()
# HighestPCAStrat.insert(0,['FligtNum','ArrivalTime','DepartureTime'])
# WriteToCsv(HighestPCAStrat,"static/ScatterMatrixStrat")

# HighestPCARand=getTop3Attributes(settings.RandFilename)
# HighestPCARand.pop(0)
# HighestPCARand = preprocessing.scale(HighestPCARand)
# HighestPCARand = HighestPCARand.tolist()
# HighestPCARand.insert(0,['FligtNum','ArrivalTime','DepartureTime'])
# WriteToCsv(HighestPCARand,"static/ScatterMatrixRand")
# return "Hello"

# if __name__== "__main__":
# app.run(host='0.0.0.0',port=5000,debug=True)