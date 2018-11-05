from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#downloading data using sklearn library and giving the present directory as folder for data backup
custom_data_home = "./"
mnist = fetch_mldata('MNIST original', data_home = custom_data_home)

#initializing num of training ,testing samples and num of samples
num_training = 6000
num_validate = 1000
num_class = 10

#storing the data into numpy array
main_data = np.array(mnist.data.shape, dtype = np.int32)
main_data = np.copy(mnist.data.astype(np.int32))

def knn():
	#k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
	#initialising k values
	k = [1,9,19,29,39,49,59,69,79,89,99]

	#shuffling the rows of data to extract unsorted training samples
	shuffle = np.arange(60000)
	np.random.shuffle(shuffle)
	
	#storing 6000 shuffked training samples and their y values
	data = np.zeros((num_training, mnist.data.shape[1]), dtype = np.int32)
	data[:] = main_data[shuffle[:num_training]]
	target = mnist.target[shuffle[:num_training]]

	#extracting and storing test samples
	validation_data = np.zeros((num_validate, mnist.data.shape[1]), dtype = np.int32)
	validation_data[:] = main_data[shuffle[num_training:(num_training+num_validate)]]
	validation_target = mnist.target[shuffle[num_training:(num_training+num_validate)]]

	#initialzing variables to store euclidean distances of training set and test set
	euclidean = np.zeros((num_training,num_training), dtype = np.int32)
	valid_euclidean = np.zeros((num_validate,num_training), dtype = np.int32)
	
	#initializing variables for storing storing sorted labels and values based on euclidean distances 
	labels = np.empty_like(euclidean)
	validate_labels = np.empty_like(valid_euclidean)

	temp = np.empty_like(euclidean)
	valid_temp = np.empty_like(valid_euclidean)

	classify = np.zeros((num_training,num_class))
	valid_classify = np.zeros((num_validate,num_class))

	#loops for calculating the euclidean distances of training and test set
	for i in range(0,num_training):
		euclidean[i:i+1, :] = np.sqrt(np.sum(np.square(data[:num_training,:] - data[i,:]),axis = 1))

	for i in range(0,num_validate):
		valid_euclidean[i:i+1, :] = np.sqrt(np.sum(np.square(data[:,:] - validation_data[i,:]),axis = 1))

	#we can uncomment the below line to export the euclidean distances into csv file
	#np.savetxt("unsorted_euclidean2.csv", euclidean, delimiter = ",")

	print("finished calculating distances")

	#getting the sorted indices of euclidean distance array
	sorted_indices = np.argsort(euclidean)
	valid_sorted_indices = np.argsort(valid_euclidean)

	#storing the labels after sorting the distances in ascending order
	labels = target[:][sorted_indices]
	validate_labels = target[:][valid_sorted_indices]

	#loops for storing distances using sorted incices
	for i in range(0,num_training):
		temp[i, :] = euclidean[i, :][sorted_indices[i]]

	for i in range(0,num_validate):
		valid_temp[i, :] = valid_euclidean[i, :][valid_sorted_indices[i]]

	euclidean[:,:] = temp[:,:]
	valid_euclidean[:,:] = valid_temp[:,:]

	np.delete(temp, np.s_[:], 1)	
	np.delete(valid_temp, np.s_[:], 1)

	#uncomment the below lines to export intermediate values to csv
	#np.savetxt("sorted_euclidean2.csv", euclidean, delimiter = ",")
	#np.savetxt("sorted_labels2.csv", labels, delimiter = ",")

	train_error = np.zeros((1,len(k)), dtype = np.float)
	test_error = np.zeros((1,len(k)), dtype = np.float)

	#loops for classifying the test and training set based on k closest distances
	for loop in k:
		for i in range(0,num_training):
			for j in range(0,loop):
				index = (int)(labels[i,j])
				classify[i,index] += 1

		for i in range(0,num_validate):
			for j in range(0,loop):
				index = (int)(validate_labels[i,j])
				valid_classify[i,index] += 1

		temp = np.argsort(classify)
		valid_temp = np.argsort(valid_classify)
		
		#loops for calculating training and test error for different k values
		for i in range(0,num_training):
			if temp[i,num_class-1] != target[i]:
				train_error[0,k.index(loop)] += 1
		
		for i in range(0,num_validate):
			if valid_temp[i,num_class-1] != validation_target[i]:
				test_error[0,k.index(loop)] += 1

		train_error[0,k.index(loop)] /= num_training
		test_error[0,k.index(loop)] /= num_validate

	# plotting the graphs on same plane for different k values and train and test errors
	plt.plot(k,train_error[0],color = 'g')
	plt.plot(k,test_error[0],color = 'orange' )
	plt.xlabel('k value')
	plt.ylabel('error')
	green_patch = mpatches.Patch(color='green', label='training error')
	orange_patch = mpatches.Patch(color='orange', label='test error')
	plt.legend(handles=[green_patch,orange_patch])
	plt.show()	
	
knn()

