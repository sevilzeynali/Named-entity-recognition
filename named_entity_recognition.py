import numpy as np
import pandas as pd
from pandas.core.common import flatten

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Model,Input
from tensorflow.keras.layers import LSTM,Embedding,Dense,Bidirectional
from tensorflow.keras import callbacks

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

import mlflow
import mlflow.keras
import mlflow.tensorflow
#this data set is available on Kaggle :https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus
df = pd.read_csv("ner_dataset.csv", encoding="latin1")
df = df.fillna(method="ffill")
df.head()

words = list(set(df["Word"].values))
print(len(words))

df = df.drop(['POS'], axis=1)
df = df.groupby('Sentence #').agg(list)
df = df.reset_index(drop=True)
df.head()

print(len(df["Word"]))

def to_1D(series):
    return pd.Series([x for liste in series for x in liste])
plt.figure(figsize=(20,10))
to_1D(df["Tag"]).value_counts().plot.bar()

words_unique=list(set(flatten(df["Word"].values)))
words_unique.append("END")
tags_unique=to_1D(df["Tag"]).value_counts().index.tolist()
print(len(words_unique))
print(len(tags_unique))

word2idx = {w : i + 1 for i ,w in enumerate(words_unique)}
tag2idx =  {t : i for i ,t in enumerate(tags_unique)}

max_len = 50
num_tags=len(tags_unique)
num_words=len(words_unique)

X = [[word2idx[w] for w in s] for s in df["Word"].values]

X = pad_sequences(maxlen = max_len, sequences = X, padding = 'post', value = num_words-1)

y = [[tag2idx[w] for w in s] for s in df["Tag"].values]
y = pad_sequences(maxlen = max_len, sequences = y, padding = 'post', value = tag2idx['O'])
y = [to_categorical(i, num_classes = num_tags) for i in  y]

x_train,x_test,y_train,y_test = train_test_split(X, y,test_size = 0.1, random_state = 1)

print(x_train.shape)
print(x_test.shape)
print(np.array(y_train).shape)
print(np.array(y_test).shape)

input_model = Input(shape=(max_len,))
embedding= Embedding(input_dim=num_words,output_dim=max_len,input_length=max_len)
lstm = Bidirectional(LSTM(units=100, activation='relu', return_sequences=True,recurrent_dropout=0.1))
dense = Dense(num_tags,activation="softmax")
x = embedding(input_model)
x = lstm(x)
output_model = dense(x)

model = Model(inputs=input_model, outputs=output_model)
model.summary()

model.compile(optimizer = 'adam',loss='categorical_crossentropy',metrics=['accuracy'])

lr_plateau = callbacks.ReduceLROnPlateau(monitor = 'loss',
                                             patience = 2,
                                             factor = 0.2,
                                             verbose = 2,
                                             mode = 'min',
                                             min_lr=0)
early_stopping = callbacks.EarlyStopping(monitor = "loss",
                                             patience = 3,
                                             mode = 'min',
                                             verbose = 2,
                                             restore_best_weights= True)
checkpoint = callbacks.ModelCheckpoint(filepath="C:/Users/Sevil/Desktop/datascientest/data_sets/check_ner",
                                          monitor = 'loss',
                                          save_best_only = True,
                                          save_weights_only = False,
                                          mode = 'min',
                                          save_freq = 'epoch')

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Named-entity recognition")
epochs=10
with mlflow.start_run() as run:    
    history = model.fit(x_train,np.array(y_train),validation_data=(x_test,y_test),validation_split = 0.2,batch_size = 32, epochs = epochs, verbose=1,use_multiprocessing= True,callbacks=[lr_plateau,early_stopping, checkpoint])
    mlflow.tensorflow.autolog(every_n_iter=1)
    mlflow.log_param("epochs",epochs)
    model_name = "Named-entity recognition"
    artifact_path="artifacts"
    mlflow.keras.log_model(keras_model=model, artifact_path=artifact_path)
    mlflow.keras.save_model(keras_model=model, path=model_name)
    mlflow.log_artifact(local_path=model_name)
    runID=run.info.run_uuid
    mlflow.register_model("runs:/"+runID+"/"+artifact_path,"ner")

train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.title("accuracy")
plt.plot(train_acc,label="train")
plt.plot(val_acc,label="test")
plt.legend()
plt.show()

train_acc = history.history['loss']
val_acc = history.history['val_loss']
plt.title("loss")
plt.plot(train_acc,label="train")
plt.plot(val_acc,label="test")
plt.legend()
plt.show()
