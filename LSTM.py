import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM,Dense
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

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



path='./data.txt'
data=np.loadtxt(path)

x=[]
y=[]
for item in data:
	x.append(item[:9])
	y.append(item[9])


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
x_train=np.array(x_train)
x_test=np.array(x_test)


n_steps_in,n_steps_out,n_features=9,1,1
x_train=x_train.reshape((x_train.shape[0],x_train.shape[1],n_features))



model=Sequential()
model.add(LSTM(50,activation='relu',return_sequences=True,input_shape=(n_steps_in,n_features)))
model.add(LSTM(50,activation='relu',return_sequences=True))
model.add(LSTM(50,activation='relu'))
model.add(Dense(n_steps_out))
model.compile(optimizer='adam',loss='mse')


net=model.fit(x_train,y_train,epochs=200,validation_split=0.2)
NetPlot(net)

x_test=x_test.reshape((x_test.shape[0],x_test.shape[1],n_features))
predicteds=model.predict(x_test)
predicteds=predicteds.flatten()


actuals=y_test
for i in range(len(predicteds)):
	print('predicted: ',predicteds[i],'actual: ',actuals[i])

print('MSE: ',mean_squared_error(actuals,predicteds))
print('RMSE: ',sqrt(mean_squared_error(actuals,predicteds)))
print('MAE: ',mean_absolute_error(actuals,predicteds))


plt.figure()
plt.plot(actuals,color='red')
plt.plot(predicteds,color='purple')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(['Actual Data','Predicted Data'])
plt.show()
