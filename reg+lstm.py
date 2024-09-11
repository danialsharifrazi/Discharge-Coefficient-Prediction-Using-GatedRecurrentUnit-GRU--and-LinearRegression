import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from keras.models import load_model
from sklearn.linear_model import LinearRegression


def NetPlot(net_history):
    history=net_history.history
    losses=history['loss']
    val_losses=history['val_loss']

    plt.figure('Loss Diagram')
    plt.title('Loss of Training Trend')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(losses)
    plt.plot(val_losses)
    plt.legend(['Training Data','Validation Data'])     
    plt.show()


path='D:/Developer/Implement/Group 2/data/data.txt'
data=np.loadtxt(path)

x=[]
y=[]
for item in data:
	x.append(item[:9])
	y.append(item[9])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train=np.array(x_train)
x_test=np.array(x_test)


model3=LinearRegression()
model3.fit(x_train,y_train)
predicteds3=model3.predict(x_test)
predicteds3=predicteds3.flatten()

x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],1))
x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],1))


model1=load_model('./model_4.h5')
model2=load_model('./model_2.h5')

predicteds1=model1.predict(x_test)
predicteds1=predicteds1.flatten()

predicteds2=model2.predict(x_test)
predicteds2=predicteds2.flatten()

predicteds=(predicteds1+predicteds3)/2


actuals=y_test
for i in range(len(predicteds)):
	print('predicted: ',predicteds[i],'actual: ',actuals[i])

print('MSE: ',mean_squared_error(actuals,predicteds))
print('RMSE: ',sqrt(mean_squared_error(actuals,predicteds)))
print('MAE: ',mean_absolute_error(actuals,predicteds))


plt.figure('fig1',dpi=200)
plt.plot(actuals,color='red')
plt.plot(predicteds,color='purple')
plt.xlabel('Data Number')
plt.ylabel('Weir')
plt.legend(['Actual Data','Predicted Data'])
plt.show()


plt.figure('fig2',dpi=200)
plt.plot(actuals,actuals,color='red')
plt.plot(actuals,predicteds,'bo',color='blue')
plt.xlabel('Observation')
plt.ylabel('Predicted')
plt.legend(['Actual Data','Predicted Data'])
plt.show()