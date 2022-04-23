import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
(X_train,y_train),(X_test,y_test)= mnist.load_data()
def plot_image(i):
  plt.imshow(X_train[i],cmap="binary")
  plt.title(y_train[i])
  plt.show()
for i in range (10):
  plot_image(i)

# NORMALIZING THE DATA
X_train=X_train.astype(np.float32)/255
X_test=X_test.astype(np.float32)/255

#INSERTS A NEW AXIS
X_train=np.expand_dims(X_train,-1)
X_test=np.expand_dims(X_test,-1)
# print(X_train.shape)
# print(X_test.shape)

# CONVERT CLASSES TO ONE HOT VECTORS
from keras.utils.np_utils import to_categorical
y_train=to_categorical(y_train)
y_test=to_categorical(y_test)
# print(y_train)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
model=Sequential()
model.add(Conv2D(32,(3,3), input_shape=(28,28,1), activation="relu"))
model.add(MaxPool2D((2,2)))

model.add(Conv2D(64,(3,3), activation="relu"))
model.add(MaxPool2D((2,2)))

model.add(Flatten())
# Dropping only 25% of nodes 
model.add(Dropout(0.25)) 
# Here, 10 is the number of classes
model.add(Dense(10,activation="softmax"))
model.summary()
# Adam is an optimization technique for gradient descent
# Combination of Gradient descent with Momentum algorithm and RMSP 
# loss is considered as cross entropy

model.compile(optimizer="adam", loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
from keras.callbacks import EarlyStopping, ModelCheckpoint
# You define and use a callback when you want to automate some tasks after every training/epoch that help you have controls over the training process. 
# This includes stopping training when you reach a certain accuracy/loss score, 
# saving your model as a checkpoint after each successful epoch, adjusting the learning rates over time

# The EarlyStoppingfunction has various metrics/arguments that you can modify to set up when the training process should stop. Here are some relevant metrics:

# monitor: value being monitored, i.e: val_loss -> decrease in loss between expected and real value
# (training should be stopped when val_acc stopped increasing -> increase in accuracy of value)
# min_delta: minimum change in the monitored value. For example, min_delta=1 means that the training process will be stopped if the absolute change of the monitored value is less than 1
# patience: number of epochs with no improvement after which training will be stopped
# restore_best_weights: set this metric to True if you want to keep the best weights once stopped
# verbose: it means to print the log after each epoch or each callback
es=EarlyStopping(monitor="val_acc", min_delta=0.01, patience=4, verbose=1)

mc=ModelCheckpoint("./bestsofar.h5",monitor="val_acc", verbose=1, save_best_only=True)
cb=[es,mc]
his=model.fit(X_train,y_train,epochs=5,validation_split=0.3,callbacks=cb)
model.save("bestsofar.h5")
model_S=keras.models.load_model("C://Users//palak sony//Desktop//Number_Detection//bestsofar.h5")
score=model_S.evaluate(X_test,y_test)
print(f"the model accuracy is{score[1]}")
