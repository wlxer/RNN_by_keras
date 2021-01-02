
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import SGD,RMSprop,Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.utils.vis_utils import plot_model

max_words = 1000
max_len = 150

#Data preprocessing
def data_preprocess():
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    df = pd.read_csv(path+'/spam.csv',delimiter=',',encoding='latin-1')
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
    X = df.v2
    Y = df.v1
    le = LabelEncoder()
    Y = le.fit_transform(Y)
    Y = Y.reshape(-1,1)
    X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
    tok = Tokenizer(num_words=max_words)
    tok.fit_on_texts(X_train)
    sequences = tok.texts_to_sequences(X_train)
    sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len) 
    test_sequences = tok.texts_to_sequences(X_test)
    test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
    return sequences_matrix,test_sequences_matrix,Y_train,Y_test

#Build RNN network
def RNN():
    # inputs = Input(name='inputs',shape=(max_len,))
    inputs = Input(name='inputs',batch_shape=(None,max_len))
    layer = Embedding(max_words,50,input_length=max_len)(inputs)
    layer = LSTM(128)(layer)
    layer = Dense(256,activation='relu',name='FC1')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,activation='sigmoid',name='out_layer')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

def main():

    sequences_matrix,test_sequences_matrix,Y_train,Y_test = data_preprocess()
    #Training
    model = RNN()
    model.summary()
    # model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
    model.compile(loss='binary_crossentropy',optimizer=Adam(),metrics=['accuracy'])
    history = model.fit(sequences_matrix,Y_train,batch_size=64,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    #Test      
    Loss, Accuracy = model.evaluate(test_sequences_matrix,Y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(Loss,Accuracy))

    #Plot acc and loss curve
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)
    plt.figure()    
    plt.title('Accuracy and Loss')
    plt.plot(epochs, acc, 'red', label='Training acc')
    plt.plot(epochs, loss, 'blue', label='Validation loss')
    plt.legend()
    # model.fit(sequences_matrix,Y_train,batch_size=16,epochs=10,
    #       validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
    # Loss, Accuracy = model.evaluate(test_sequences_matrix,Y_test)
    # print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(Loss,Accuracy))

    #Show specific layers of RNN
    plot_model(model, to_file="model.png",show_shapes=True,show_layer_names=False,rankdir='TB')
    plt.figure(figsize=(10,10))
    img = plt.imread("model.png")
    plt.imshow(img)
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()