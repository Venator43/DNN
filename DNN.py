import numpy as np
from random import random

class MLP:
	
	def __init__(self,input_layer=3,hidden_layer=[2,3],output_layer=4):
		self.num_il = input_layer
		self.num_hl = hidden_layer
		self.num_ol = output_layer

		layer = [self.num_il] + self.num_hl + [self.num_ol]
				
		self.weight = []
		#For every number of node in the network output a random value for a weight in it
		#this will create a weight in every node that will connect to the next node in the next layer
		for i in range(len(layer)-1):
			w = np.random.rand(layer[i],layer[i+1])
			self.weight.append(w)
		
		#create a list to store all the activation layer, the list has a shape of matrix with value of 0 and the size of those respected layer
		activation = []	
		for i in range(len(layer)):
			activation.append(np.zeros(layer[i]))
		self.actv_layer = activation
		
		#create a list to store all the derivative of the weight, the list has a shape of matrix with value of 0 and the size of the weight of those respected layer
		self.derivative = []	
		for i in range(len(layer)-1):
			self.derivative.append(np.zeros((layer[i],layer[i+1])))

	def fit(self, items, targets, epochs, l_rate):
		#epoch is how many cycle you want the neural network to do the forward propagation and backward propagation per data
		for i in range(epochs):
			accuracy = 0

			for item,target in zip(items, targets):
				#First we put our data to the network our data of which consist of the input for the input layer and the expected prediction for that input
				out = self.forward(item)
				#Next we count how much our weight made an error by comparing our network prediction with our expected output
				error = target - out
				#Using that error value we then backward propagate it so we can get the derivative of every weight in our neural network
				self.backward(error)
				#Then we use gradient decent in order to get the global minimum value for our weight and change our original weight to that value
				self.grad_dec(l_rate)
				#Count the acuracy of the neural network
				accuracy += self.mse(target,out)

			print(f"Error {accuracy/len(items)} at epoch {i}")

		print("\nTraining Completed")

	def mse(self,target,out):
		return np.average((target - out) ** 2)			

	def _sigmoid(x):
		return 1/(1+np.exp(-x))

	def forward(self,input,act_func=_sigmoid):
		activation = input
		self.actv_layer[0] = input

		for i,w in enumerate(self.weight):
			#Calculate layer dot value
			net_input = np.dot(activation,w)
			#Get next layer activated value
			activation = act_func(net_input)

			self.actv_layer[i+1] = activation

		return activation

	def backward(self,error,Verbose=False):
		#Using chain rule a formula to count the error value of each layer was found
		#dE/dW_i = (y-a_[i+1]) * s'(h_[i+1]) * a_i
		#s'(h_[i+1]) = s_(h_[i+1])(1-s(h[i+1]))
		#s(h_[i+1]) = a_[i+1]
		#Since we already knew the activation value we can just plug the value into the formula
		#error = (y-a_[i+1]) 

		for i in reversed(range(len(self.derivative))):
			delta = error * self.sig_derv(self.actv_layer[i+1])
			delta_r = delta.reshape(delta.shape[0], -1).T #Reshape the array into 2-D array (from [1,2,3,4] into [[1,2,3,4]])
			cur_act = self.actv_layer[i]
			cur_act = cur_act.reshape(cur_act.shape[0],-1) #Reshape the array into 2-D array (from [1,2,3,4] into [[1],[2],[3],[4]]])

			self.derivative[i] = np.dot(cur_act,delta_r)
			#To get the derivative of the previous error function of the previous layer we will used the formula:
			#dE/dW_1 = (y-a_[i+1]) * s'(h_[i+1]) * w_i * s'(h_i) * a_[i-1]
			#Notice that we already count (y-a_[i+1]) * s'(h_[i+1]) and store it in the variable delta
			#And since the rest of the formula is basically the same we can just change the value of the error with the dot product of the delta with the weight
			error = np.dot(delta, self.weight[i].T)

			if Verbose:
				print(f"The error derivative of {i}-Layer is: {self.derivative[i]}")

		return error

	def sig_derv(self,x):
		return x * (1.0 - x)

	def grad_dec(self,l_rate):
		for i in range(len(self.weight)):
			weight = self.weight[i]
			derivative = self.derivative[i]
			#print(f"Current W_{i} : {weight}")
			weight += derivative * l_rate
			#print(f"Chaged W_{i} : {weight}")


#Initiate MLP
NN = MLP(2,[10,10],1)

#Create a dummy data NOTE node value can only be in between of 0 and 1 with the value of the node represent how "active" the node is
item = np.array([[random()/2 for _ in range(2)] for _ in range(1000)])
target = np.array([[i[0] + i[1]] for i in item])

#Train the MLP
NN.fit(item,target,50,1)

#Predict a new data
print(f"The network belive that the value of {0.1} + {0.2} is equal to {NN.forward([0.1,0.2])}")