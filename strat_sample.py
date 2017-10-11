import app
from sklearn.cluster import KMeans
import settings
import csv
import random

def readData(filename):
	with open(filename,"r") as fd:
		reader=csv.reader(fd)
		headers=next(reader)
		list_data=list(reader)
	return list_data,headers


def StringtoInt(list_data):
	List_Data=[]
	for l in list_data:
		Li=[]
		for i in range(len(l)):
			Li.append(float(l[i]))
		List_Data.append(Li)
	return List_Data


def MakeClusters(list_data,n_clusters):
	ClustersDict={}
	kmeans=KMeans(n_clusters=n_clusters , random_state=0).fit(list_data)
	for i in range(n_clusters):
		ClustersDict[i]=[]
	labels=kmeans.labels_
	for i in range(len(labels)):
		ClustersDict[labels[i]].append(list_data[i])
	return ClustersDict,kmeans.cluster_centers_

def StratifiedSample(ClustersDict):
	StratSample=[]
	for i in range(len(ClustersDict)):
		StratSample.extend(RandomSampling(ClustersDict[i]))
	return StratSample

def RandomSampling(list_data):
	rand_smpl=[list_data[i] for i in random.sample(range(len(list_data)),int(settings.samplePercent*len(list_data)))]
	return rand_smpl

def WriteToCsv(data,filename):
	with open(filename+'.csv',"w") as fd:
		wr=csv.writer(fd)
		wr.writerows(data)

list_data,headers=readData('six.csv')

list_data=StringtoInt(list_data)
# SSEPlotData=[]
# for i,row in enumerate(SSE):
# 	SSEPlotData.append([i,SSE[i]])
# SSEPlotData.insert(1,['i','ev'])
# WriteToCsv(SSEPlotData[1:],'static/ElbowPlot')
optK=3

ClustersDict,ClusterCentres=MakeClusters(list_data,optK)

StratSampleData=StratifiedSample(ClustersDict)
StratSampleData.insert(0,headers)
WriteToCsv(StratSampleData,'static/cars')