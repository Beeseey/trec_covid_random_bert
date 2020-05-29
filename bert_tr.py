from keras_bert import load_trained_model_from_checkpoint
from bert.tokenization import bert_tokenization
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout, Flatten, RepeatVector, TimeDistributed, Bidirectional, concatenate, Lambda, dot, Activation
from tensorflow.keras.layers import add
from tensorflow.keras.optimizers import Adam
import keras
from parse_data_bert_3 import get_data
import os
import random

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def createTokenizer():
	currentDir = os.path.dirname(os.path.realpath(__file__))
	modelsFolder = os.path.join(currentDir, "models", "multi_cased_L-12_H-768_A-12")
	vocab_file = os.path.join(modelsFolder, "vocab.txt")

	tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
	return tokenizer

#X,X1,Y = get_data()

data = get_data()
random.shuffle(data)
X = []
X1 = []
Y = []

for d in data:
	X.append(d[0])
	X1.append(d[1])
	Y.append(d[2])

'''
tokens = [tokenizer.tokenize(toks) for toks in inputs]
tokens = [tokenizer.convert_tokens_to_ids(toks) for toks in tokens]
tokens = [toks + [0]*(max_len - len(toks)) for toks in tokens]
'''

dataset_len = len(X)
train_X = X[:int(dataset_len*0.85)]
train_X1 = X1[:int(dataset_len*0.85)]
train_Y = Y[:int(dataset_len*0.85)]
'''
train_X = X[100:1000]
train_X1 = X1[100:1000]
train_Y = Y[100:1000]
'''
'''
for val in train_X:
	print(val)
raise('stop')
'''

test_X = X[int(dataset_len*0.85):]
test_X1 = X1[int(dataset_len*0.85):]
test_Y = Y[int(dataset_len*0.85):]
print(len(test_X),len(train_Y),len(train_X),len(test_Y))
'''
test_X = X[:100]
test_X1 = X1[:100]
test_Y = Y[:100]
'''
def get_max_seq(X):
	return max([len(x) for x in X])
modelsFolder = os.path.join('models', "multi_cased_L-12_H-768_A-12")
checkpointName = os.path.join(modelsFolder, "bert_model.ckpt")

def get_segment_idx(token_ids):
	token_segment = []
	a = 0
	for idx in token_ids:
		if idx == 0:
			a = 0
		token_segment.append(a)
		if idx == 102:
			a = 1


	segment_idx = np.array(token_segment)
	return segment_idx
tokenizer = createTokenizer()
#print('Restructuring training')
#tokenizer = createTokenizer()
#trainX_tokens = map(tokenizer.tokenize,train_X)
#trainX1_tokens = map(tokenizer.tokenize,train_X1)
print('creating train tokens')
#train_tokens = map(lambda tok1,tok2: ["[CLS]"] + tok1 + ["[SEP]"] + tok2 + ["[SEP]"], train_X,train_X1)
train_tokens = []

for tok1,tok2 in zip(train_X,train_X1):
	train_tokens.append(tokenizer.convert_tokens_to_ids(["[CLS]"] + tok1 + ["[SEP]"] + tok2 + ["[SEP]"]))
#train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

train_segment_ids = get_segment_idx(train_tokens)

max_len = get_max_seq(train_tokens)
print(max_len)

#train_token_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), train_tokens)
train_token_ids = []

for tids in train_tokens:
	train_token_ids.append(tids + [0] * (max_len - len(tids)))

#train_segement_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), train_segment_ids)
#train_token_ids = np.array(list(train_token_ids))

#print('Restructuring testing')
#testX_tokens = map(tokenizer.tokenize,test_X)
#testX1_tokens = map(tokenizer.tokenize,test_X1)
print('creating test tokens')
test_tokens = []

for tok1,tok2 in zip(test_X,test_X1):
	test_tokens.append(tokenizer.convert_tokens_to_ids(["[CLS]"] + tok1 + ["[SEP]"] + tok2 + ["[SEP]"]))

#test_tokens = map(lambda tok1,tok2: ["[CLS]"] + tok1 + ["[SEP]"] + tok2 + ["[SEP]"], test_X,test_X1)

#test_token_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))

test_token_ids = []
for tids in test_tokens:
	test_token_ids.append(tids + [0] * (max_len - len(tids)))
test_segment_ids = get_segment_idx(test_token_ids)
#test_token_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), test_tokens)
#test_segement_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), test_segment_ids)
train_x = train_token_ids
test_x = test_token_ids
print(len(train_x),len(test_x))
print(len(list(test_tokens)))
print(len(list(train_tokens)))

bert_ = load_trained_model_from_checkpoint(
      "models/multi_cased_L-12_H-768_A-12/bert_config.json",
      checkpointName,
      training=True,
      trainable=True,
      seq_len=max_len)

#print(bert_.summary())

def bert_mod(model):
	for layer in model.layers:
		layer.trainable = False 
	inputs = model.inputs
	dense = model.get_layer('Extract').output
	#print(dense)
	dense = keras.layers.Dropout(0.5)(dense)
	dense = keras.layers.Dense(256)(dense)
	dense = keras.layers.Dropout(0.5)(dense)
	dense = keras.layers.Dense(256)(dense)
	dense = keras.layers.Dropout(0.5)(dense)
	outputs = keras.layers.Dense(units=3, activation='softmax')(dense)

	model = keras.models.Model([model.inputs[0],model.inputs[1]], outputs)
	model.compile(
  		optimizer='adam',
  		loss='sparse_categorical_crossentropy',
  		metrics=['sparse_categorical_accuracy'])

	return model

def batch_generator(ids,labels,batch_size):
	count = 0
	while True:
		for start in range(0, len(ids), batch_size):
			x = ids
			y = labels
			x_batch = []
			x1_batch = []
			y_batch = []
			end = min(start + batch_size, len(x))
			ids_batch = x[start:end]
			label_batch = y[start:end]
			for id_,label in zip(ids_batch,label_batch):
				count+=1
				seg_id = get_segment_idx(id_)
				#print(count,'llll')
				x_batch.append(id_)
				x1_batch.append(seg_id)
				y_batch.append(label)
				if len(x_batch) == batch_size:
					b = np.array(x_batch)
					b1 = np.array(x1_batch)
					d = np.array(y_batch)
					#print(len(b))
					#print(len(d))
					yield [b,b1],d 

model = bert_mod(bert_)

#print(model.summary())



cp_callback = keras.callbacks.ModelCheckpoint(filepath='bert.model',
	monitor='val_loss', verbose=1, save_best_only=True, mode='min')




batch_size = 50

train_gen = batch_generator(train_x,train_Y,batch_size)
test_gen = batch_generator(test_x,test_Y,batch_size)

model.fit(train_gen,epochs=100,verbose=1,steps_per_epoch=len(train_x)//batch_size,
	validation_data=test_gen,validation_steps=len(test_x)//batch_size,callbacks=[cp_callback])