
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt


plt.switch_backend('TkAgg')  

# Initialize two empty lists
dates = []
prices = []

def get_data(filename):
	'''
	Reads data from a file (snap.csv) and adds data to
	the lists dates and prices
	'''
	# Use the with as block to open the file and assign it to csvfile 
	with open(filename, 'r') as csvfile:
		# csvFileReader allows us to iterate over every row in our csv file
		csvFileReader = csv.reader(csvfile)
		next(csvFileReader)	# skipping column names
		for row in csvFileReader:
			dates.append(int(row[0].split('-')[0])) # Only gets day of the month which is at index 0
			prices.append(float(row[1])) # Convert to float for more precision

	return

def predict_price(dates, prices, x):
	print("dates"+str(dates))
	print("prices="+str(prices))
	print("x"+str(x))

	dates = np.reshape(dates,(len(dates), 1)) # converting to matrix of n X 1
	print("dates1="+str(dates))

	# If both are right on top of each other, the max similarity is one, if too far it is a ze
	svr_lin = SVR(kernel= 'linear', C= 1e3) # 1e3 denotes 1000
	svr_poly = SVR(kernel= 'poly', C= 1e3, degree= 2)
	svr_rbf = SVR(kernel= 'rbf', C= 1e3, gamma= 0.1) # defining the support vector regression models
	
	svr_rbf.fit(dates, prices) # fitting the data points in the models
	svr_lin.fit(dates, prices)
	svr_poly.fit(dates, prices)



	plt.scatter(dates, prices, color= 'black', label= 'Data') # plotting the initial datapoints 

	plt.plot(dates, svr_rbf.predict(dates), color= 'red', label= 'RBF model') # plotting the line made by the RBF kernel
	plt.plot(dates,svr_lin.predict(dates), color= 'green', label= 'Linear model') # plotting the line made by linear kernel
	plt.plot(dates,svr_poly.predict(dates), color= 'blue', label= 'Polynomial model') # plotting the line made by polynomial kernel
	plt.xlabel('Date') # Setting the x-axis
	plt.ylabel('Price') # Setting the y-axis
	plt.title('Support Vector Regression') # Setting title
	# plt.legend() # Add legend
	# plt.show() # To display result on screen
	print(svr_rbf.predict( np.reshape([x],(len([x]), 1)))[0])
	return svr_rbf.predict(np.reshape([x],(len([x]), 1)))[0], svr_lin.predict(np.reshape([x],(len([x]), 1)))[0], svr_poly.predict(np.reshape([x],(len([x]), 1)))[0] # returns predictions from each of our models

get_data('snap.csv') # calling get_data method by passing the csv file to it

predicted_price = predict_price(dates, prices, 18)

print('The predicted prices are:', predicted_price)
