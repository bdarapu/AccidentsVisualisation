def readDataColumns(filename):
	list_dataColumns=[]
	with open(filename,"r") as fd:
		reader=csv.reader(fd)
		headers=next(reader)
		for i in range(len(headers)):
			dict1={}
			dict1[headers[i]]={}
			list_dataColumns.append(dict1[headers[i]])
		for row in reader:
			for i in range(len(row)):
				dict1[headers[i]].append(row[i])
	# for i in range(len(headers)):
	# 	list_dataColumns.append(dict1[headers[i]])
	return list_dataColumns



# Read the data as list of dictionaries - not using as of now
def readDataDict(filename):
	list_data = []
	with open(filename, "r") as infile:
	    reader = csv.reader(infile)
	    headers = next(reader)[:]
	    for row in reader:
	    	dict1={}
	    	for i in range(len(row)):
	    		if isinstance(row[i],int):
	    			dict1[headers[i]]=int(row[i]) 
	    		else:
	    			dict1[headers[i]]=row[i]
	    	list_data.append(dict1)
	return list_data


dict_dataColumns=readDataColumns(settings.filename)
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

	
	eigenValues=LA.eigvals(list_dataCovariance)

	X=[]
	for i in range(len(eigenValues)):
		X.append(i)

	plt.plot(X,eigenValues)
	plt.show()

	dataVariance=ComputeVariance(dict_dataColumns)

	print("three attributes with highest PCA loadings: ",dataVariance[:3])
	


var w=500;
 var h=350;
 // var dataset=[10,20,25];
 var barPadding=5;	
 var dataset1=[]
 var X=[]
 var Y=[]

 d3.csv("screePlotRand.csv",function(datum){
 	datum.forEach(function (d){
 		d.i=+d.i;
 		d.ev=+d.ev;
 	})
 })