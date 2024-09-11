from sklearn.linear_model import LogisticRegression
from keras.models import load_model
import numpy as np
from numpy import dstack
from sklearn.model_selection import train_test_split

path='./data.txt'
data=np.loadtxt(path)

x=[]
y=[]
for item in data:
	x.append(item[:9])
	y.append(item[9])

trainX,testX,trainy,testy=train_test_split(x,y,test_size=0.2,random_state=0)
trainX=np.array(trainX)
testX=np.array(testX)
trainX=trainX.reshape((trainX.shape[0],trainX.shape[1],1))
testX=testX.reshape((testX.shape[0],testX.shape[1],1))



# load models from file
def load_all_models(n_models):
	all_models = list()
	for i in range(n_models):
		filename = 'models/model_' + str(i + 1) + '.h5'
		model = load_model(filename)
		all_models.append(model)
		print('>loaded %s' % filename)
	return all_models

def stacked_dataset(members, inputX):
	stackX = None
	for model in members:
		yhat = model.predict(inputX)
		if stackX is None:
			stackX = yhat
		else:
			stackX = dstack((stackX, yhat))
	stackX=stackX.reshape((stackX.shape[0],stackX.shape[1]*stackX.shape[2],1))
	return stackX

def fit_stacked_model(members, inputX, inputy):
	stackedX = stacked_dataset(members, inputX)
	inputy=np.array(inputy)
	model = LogisticRegression()
	stackedX=stackedX.reshape(24,2)
	stackedX=stackedX[:24]
	print('our shape1 is: ',stackedX.shape)
	print('our shape2 is: ',inputy.shape)
	model.fit(stackedX, inputy)
	return model

def stacked_prediction(members, model, inputX):
	stackedX = stacked_dataset(members, inputX)
	yhat = model.predict(stackedX)
	return yhat


# load all models
n_members = 2
members = load_all_models(n_members)
print('Loaded %d models' % len(members))
model = fit_stacked_model(members, testX, testy)



