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

app = Flask(__name__)

@app.route("/")
def index():
	return render_template("index.html")

@app.context_processor
def override_url_for():
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)

@app.route("/data")
def dataProcessing():
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
		return sorted_dataVariance

	def WriteToCsv(data,filename):
		with open(filename+'.csv',"w") as fd:
			wr=csv.writer(fd)
			wr.writerows(data)

	def getTopAttributes(filename):
		dict_dataColumns=readDataColumns(filename)
		dataVariance=ComputeVariance(dict_dataColumns)
		top3_attributes=[]
		for i in range(len(dataVariance)):
			top3_attributes.append(dataVariance[i][0])
		# print(top3_attributes)

		listTopCol=[]
		for key in top3_attributes:
			listTopCol.append(dict_dataColumns[key])
		return listTopCol

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
		

		for i,k in enumerate(list_columnorder):
			dict_dataCovariance[k]=list_dataCovariance[i]

		listTopCol=getTopAttributes(filename)
		eigenValues=LA.eigvals(list_dataCovariance)
		# eigenValuesCorr=LA.eigvals(list_dataCorelation)

		screePlot=[]
		screePlot.append(['i','ev'])
		for i,ev in enumerate(eigenValues):
			li=[i,ev]
			screePlot.append(li)
		sum_eig=sum(eigenValues)
		TopLoadings = []
		TopLoadings.append(['i','ev'])
		for i in range(len(eigenValues)):
			li=[list_columnorder[i],eigenValues[i]/sum_eig]
			TopLoadings.append(li)

		return screePlot , TopLoadings
		
	#trying with list_data from sampled data as input and directly
	#passing it to PCA ;dicarding top3 attributes calculated by me
	def PCAdata(filename):
		

		list_data , _ = readData(filename)
		list_data = StringtoInt(list_data)

		pca = PCA(n_components=2)
		pcaNumpy=pca.fit_transform(list_data)
		pcaList=[]
		for i in range(len(pcaNumpy)):
			x=float(pcaNumpy[i][0])
			y=float(pcaNumpy[i][1])
			li=[x,y]
			pcaList.append(li)
		return pcaList

	
	#taking the random samplified data as input to pairwise_distance
	
	def MDSdata(filename,type_mds,list_data):
		
		dis_mat=pairwise_distances(list_data,metric=type_mds)
		mds=MDS(n_components=2,dissimilarity='precomputed')
		mdsNumpy=mds.fit_transform(dis_mat)
		
		mdsList=[]
		for i in range(len(mdsNumpy)):
			x=float(mdsNumpy[i][0])
			y=float(mdsNumpy[i][1])
			li=[x,y]
			mdsList.append(li)
		return mdsList

	

		# print(listTop3)
		# listTop3=[]
		# for i in range(len(listTop3Col[0])):
		# 	li=[]
		# 	for j in range(len(listTop3Col)):
		# 		li.append(listTop3Col[j][i])
		# 	listTop3.append(li)
		# listTop3.insert(0,top3_attributes)
		# return listTop3

	def PCA3(filename):
		list_data , _ = readData(filename)
		list_data = StringtoInt(list_data)
		pca = PCA(n_components=3)
		pcaNumpy=pca.fit_transform(list_data)
		pcaList=[]
		for i in range(len(pcaNumpy)):
			x=float(pcaNumpy[i][0])
			y=float(pcaNumpy[i][1])
			z=float(pcaNumpy[i][2])
			li=[x,y,z]
			pcaList.append(li)
		# pcaList.insert(0,['FligtNum','ArrivalTime','DepartureTime'])
		return pcaList

	
	list_data,headers=readData(settings.filename)

	list_data=StringtoInt(list_data)
	list_data = preprocessing.scale(list_data)

	randSampleData=RandomSampling(list_data)
	
	randSampleData.insert(0,headers)
	WriteToCsv(randSampleData,'static/RandomSamplingData')

	SSE=[]
	SSE.append(0)
	for i in range(1,settings.maxK):
		ClustersDict,ClusterCentres=MakeClusters(list_data,i)
		SSE.append(GetSSE(ClustersDict,ClusterCentres))

	
	SSEPlotData=[]
	for i,row in enumerate(SSE):
		SSEPlotData.append([i,SSE[i]])
	SSEPlotData.insert(1,['i','ev'])
	WriteToCsv(SSEPlotData[1:],'static/ElbowPlot')
	optK=3

	ClustersDict,ClusterCentres=MakeClusters(list_data,optK)

	StratSampleData=StratifiedSample(ClustersDict)
	StratSampleData.insert(0,headers)
	WriteToCsv(StratSampleData,'static/StratifiedSamplingData')

	screePlotRand , TopLoadingsRand =DimensionReduction(settings.RandFilename)
	WriteToCsv(screePlotRand,'static/screePlotRand')
	WriteToCsv(TopLoadingsRand,'static/TopLoadingsRand')

	

	screePlotStrat , TopLoadingsStrat=DimensionReduction(settings.StratFilename)
	WriteToCsv(screePlotStrat,'static/screePlotStrat')
	WriteToCsv(TopLoadingsStrat,'static/TopLoadingsStrat')

	listScatterStrat=[]
	listScatterStrat=PCAdata(settings.StratFilename)
	listScatterStrat.insert(0,['a1','a2'])
	WriteToCsv(listScatterStrat,"static/PCAPlotStrat")
	

		
	listScatterRand=[]
	listScatterRand=PCAdata(settings.RandFilename)
	listScatterRand.insert(0,['a1','a2'])
	WriteToCsv(listScatterRand,"static/PCAPlotRand")

	
	listScatterStratMds_euclidean=MDSdata(settings.StratFilename,"euclidean",StratSampleData[1:])
	listScatterStratMds_euclidean.insert(0,['a1','a2'])
	WriteToCsv(listScatterStratMds_euclidean,"static/StratMdsEuclidean")

	#passing the whole data instead of the top3 attributes
	listScatterStratMds_correlation=MDSdata(settings.StratFilename,"cosine",StratSampleData[1:])
	listScatterStratMds_correlation.insert(0,['a1','a2'])
	WriteToCsv(listScatterStratMds_correlation,"static/StratMdscorrelation")

	listScatterRandMds_euclidean=MDSdata(settings.RandFilename,"euclidean",randSampleData[1:])
	listScatterRandMds_euclidean.insert(0,['a1','a2'])
	WriteToCsv(listScatterRandMds_euclidean,"static/RandMdsEuclidean")

	listScatterRandMds_correlation=MDSdata(settings.RandFilename,"cosine",randSampleData[1:])
	listScatterRandMds_correlation.insert(0,['a1','a2'])
	WriteToCsv(listScatterRandMds_correlation,"static/RandMdsCorrelation")

	
	HighestPCAStrat = PCA3(settings.StratFilename)
	HighestPCAStrat.insert(0,[TopLoadingsStrat[1][0],TopLoadingsStrat[2][0],TopLoadingsStrat[3][0]])
	WriteToCsv(HighestPCAStrat,"static/ScatterMatrixStrat")

	HighestPCARand=PCA3(settings.RandFilename)
	HighestPCARand.insert(0,[TopLoadingsRand[1][0],TopLoadingsRand[2][0],TopLoadingsRand[3][0]])
	WriteToCsv(HighestPCARand,"static/ScatterMatrixRand")
	return "Hello"

if __name__== "__main__":
	app.run(host='0.0.0.0',port=5000,debug=True)