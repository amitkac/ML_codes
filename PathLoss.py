# project

from sklearn.preprocessing import LabelBinarizer,StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import sys 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential # means the network will be feedforward
from keras.layers.core import Dense  # implementation of fully connected layer

from keras.optimizers import SGD    # use SGD to optimize the network
time=300

freq_df=pd.read_csv('C:/Users/Amit/Documents/PythonTutorial/CodeD/test.csv',header=None,names=['phase','d','pathloss','env'])
freq_df['pathloss']=-1*freq_df['pathloss']
#freq_df['d'].astype(float)


one_hot = pd.get_dummies(freq_df['env'])
freq_df = freq_df.drop('env', axis=1)
freq_df = freq_df.join(one_hot)
#lb=LabelEncoder()
#print(freq_df)
sc=MinMaxScaler(feature_range = (0,1))
freq_df[['phase','d']]=sc.fit_transform(freq_df[['phase','d']])
#df[df.columns.difference(['b'])]
print(freq_df.head())
# sys.exit(0)


target=freq_df['pathloss']

inputD=freq_df.drop(columns = ['pathloss'], axis = 1)
(trainX,testX,trainY,testY)=train_test_split(inputD,target,test_size=0.25)

#sc=MinMaxScaler()
#sc.fit_transform(trainX,testX,trainY,testY)
#print(testY)

model=Sequential()
model.add(Dense(6,input_shape=(6,),kernel_initializer='normal',activation="relu"))
model.add(Dense(6,kernel_initializer='normal',activation="relu"))
model.add(Dense(3,kernel_initializer='normal',activation="relu"))
model.add(Dense(1,kernel_initializer='normal'))

sgd=SGD(0.001)
opt=SGD(lr=0.01,decay=0.01 / 40, momentum=0.9,nesterov=True)
model.compile(loss='mse', optimizer=opt, metrics=['mse']) # use cross entropy as loss metric and the reason to change the int to vector
#loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', 'cosine']
H=model.fit(trainX,trainY,validation_data=(testX,testY),epochs=time) # remember batch size makes it the back propagation 
# In reality it is a wrong practice to use test data as validation data---it has to be different than test and training which the model has not seen before.
print(model.summary())

# network evaluation
print("Evaluating network")
predictions=model.predict(testX,batch_size=32)
#pred=predictions.flatten()
pred=predictions[0][0]
#print(np.sum((testY-pred)**2))
#for layer in model.layers:
    #w=layer.get_weights()
    #con=layer.get_config()
    #print(w)
    
#print(con)


plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,time),H.history['mean_squared_error'], label="mse")
plt.plot(np.arange(0,time), H.history["loss"], label="train_loss")
#plt.plot(np.arange(0,time),H.history['mean_absolute_error'])
#plt.plot(np.arange(0,time),H.history['mean_absolute_percentage_error'])
#plt.plot(np.arange(0,time),H.history['cosine_proximity'])
plt.title("Training loss and accuracy")
plt.xlabel("Epoch");
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()