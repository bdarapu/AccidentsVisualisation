from flask import Flask
import csv




app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World November Rain!"

@app.route("/readData")
def readData():
	dict1 = {}
	with open("data.csv", "r") as infile:
	    reader = csv.reader(infile)
	    headers = next(reader)[1:]
	    for row in reader:
	        dict1[row[0]] = {key: value for key, value in zip(headers, row[1:])}
	    return "summer"
	    print(dict1)

if __name__ == "__main__":
    app.run(debug=True)