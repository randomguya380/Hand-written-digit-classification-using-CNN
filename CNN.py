import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import
dataset = pd.read_csv('digitd.csv')
X = dataset.iloc[:, 0:256].values
y = dataset.iloc[:, 256:266].values
 
#split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

#scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#visualising
'''d=X[160,:].reshape(16,16)
plt.imshow(d);
plt.show()'''

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

epoch=6000
lr=0.005 
input_neurons = X.shape[1] 
hidden_neurons = 26 #number of hidden layers neurons
output_neurons = 10 #number of neurons at output layer

#weight and bias initialization
w1=np.random.uniform(size=(input_neurons,hidden_neurons))
b1=np.random.uniform(size=(1,hidden_neurons))
w2=np.random.uniform(size=(hidden_neurons,output_neurons))
b2=np.random.uniform(size=(1,output_neurons))
error=[]

for i in range(epoch):

#Forward Propogation
    z2= np.dot(X_train,w1)
    z2= z2 + b1
    a2 = sigmoid(z2)
    z3=np.dot(a2,w2)
    z3= z3+ b2
    a3 = sigmoid(z3)
    e=(1/(2*1592))*(np.sum(pow((y_train-a3),2)))
    print(i,e)
    error.append(e)
    
    if e < 0.03 : break
    
    

#Backpropagation
    E = y_train-a3
    slope_output_layer = derivatives_sigmoid(a3)
    slope_hidden_layer = derivatives_sigmoid(a2)
    delta3 = E * slope_output_layer
    Error_at_hidden_layer = delta3.dot(w2.T)
    delta2 = Error_at_hidden_layer * slope_hidden_layer
    
    w2 += a2.T.dot(delta3) *lr
    b2+= np.sum(delta3, axis=0,keepdims=True) *lr
    w1 += X_train.T.dot(delta2) *lr
    b1 += np.sum(delta2, axis=0,keepdims=True) *lr

'''plt.scatter(range(epoch),error)
plt.show()'''

z2=np.dot(X_test,w1)
z2=z2 + b1
a2 = sigmoid(z2)
z3=np.dot(a2,w2)
z3= z3+ b2
a3 = sigmoid(z3)

def predict(a2):
    k=np.argmax(a2,axis=1)
    return k
y_pre=predict(a3)
y_true=predict(y_test)

a=np.sum(y_pre==y_true)
print("accuracy :" ,(a/478)*100)

z2=np.dot(X_train,w1)
z2=z2 + b1
a2 = sigmoid(z2)
z3=np.dot(a2,w2)
z3= z3+ b2
a3 = sigmoid(z3)


y_pre=predict(a3)
y_true=predict(y_train)

a=np.sum(y_pre==y_true)
print("accuracy :" ,(a/1114)*100)


