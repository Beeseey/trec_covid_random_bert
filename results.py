from keras_bert import load_trained_model_from_checkpoint
from bert.tokenization import bert_tokenization
import tensorflow as tf
import numpy as np
import keras
#from keras.models import load_weights
import os 
from parse_data_bert import get_data_

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

def createTokenizer():
	currentDir = os.path.dirname(os.path.realpath(__file__))
	modelsFolder = os.path.join(currentDir, "models", "multi_cased_L-12_H-768_A-12")
	vocab_file = os.path.join(modelsFolder, "vocab.txt")

	tokenizer = bert_tokenization.FullTokenizer(vocab_file, do_lower_case=True)
	return tokenizer

#X,X1= get_data_()

def get_max_seq(X):
	return max([len(x) for x in X])

def get_segment_idx(token_ids):
	
	token_segment = []
	a = 0
	for idx in token_ids:
		token_segment.append(a)
		if idx == 102:
			a = 1
	return token_segment

max_len = 428

modelsFolder = os.path.join('models', "multi_cased_L-12_H-768_A-12")
checkpointName = os.path.join(modelsFolder, "bert_model.ckpt")

bert_ = load_trained_model_from_checkpoint(
      "models/multi_cased_L-12_H-768_A-12/bert_config.json",
      checkpointName,
      training=True,
      trainable=True,
      seq_len=max_len)

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

model = bert_mod(bert_)
print(model.summary())
model.load_weights('bert.model')


X = get_data_()
tokenizer = createTokenizer()

file = open('corpus2/r2_resultsb.txt', 'w+',encoding='utf-8')

count = 0
for x_dict in X:

	print(count,'out of',len(X))
	count+=1
	x = x_dict['X']
	x1 = x_dict['X1']
	cord_id = x_dict['cord_id']
	query = x_dict['query']
	title = x_dict['title']
	x_tokens = tokenizer.tokenize(x)
	x1_tokens = tokenizer.tokenize(x1)

	if len(x_tokens)+len(x1_tokens)-3 >420:
		diff = len(x_tokens)+len(x1_tokens)+3 - 420
		x1_tokens = x1_tokens[:len(x1_tokens)-diff]

	token = ["[CLS]"]+x_tokens+["[SEP]"]+x1_tokens+["[SEP]"]
	tids = tokenizer.convert_tokens_to_ids(token)
	if len(tids) > 428:
		tids = tids[:428]
	segment_ids = get_segment_idx(tids)
	token_ids = tids + [0] * (max_len - len(tids))
	segment_ids = segment_ids + [0] * (max_len - len(segment_ids))
	#print(np.array(token_ids).shape,np.array(segment_ids).shape)
	c = np.array(token_ids)
	b = np.array(segment_ids)
	c = np.expand_dims(c,axis=0)
	b = np.expand_dims(b,axis=0)
	print(c.shape,b.shape)
	result = model.predict([c,b])[0]
	classi = list(result).index(max(result))
	print(result)
	if classi == 1 or classi == 2:
		text = str(query)+' '+'Q0'+' '+cord_id+' '+str(classi)+' '+str(result[classi])+' '+'random_bert_titleabstract'
		file.write(text+"\n")

file.close()



#weights_list = model.get_weights()
'''
print('Restructuring training')
tokenizer = createTokenizer()
trainX_tokens = map(tokenizer.tokenize,train_X)
trainX1_tokens = map(tokenizer.tokenize,train_X1)

train_tokens = map(lambda tok1,tok2: ["[CLS]"] + tok1 + ["[SEP]"] + tok2 + ["[SEP]"], trainX_tokens,trainX1_tokens)
train_token_ids = list(map(tokenizer.convert_tokens_to_ids, train_tokens))

train_segment_ids = get_segment_idx(train_token_ids)

print(train_segment_ids[0])

#raise('stop')



max_len = get_max_seq(train_token_ids)
print(max_len)

train_token_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), train_token_ids)
train_segement_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), train_segment_ids)
#train_token_ids = np.array(list(train_token_ids))

print('Restructuring testing')
testX_tokens = map(tokenizer.tokenize,test_X)
testX1_tokens = map(tokenizer.tokenize,test_X1)

test_tokens = map(lambda tok1,tok2: ["[CLS]"] + tok1 + ["[SEP]"] + tok2 + ["[SEP]"], testX_tokens,testX1_tokens)
test_token_ids = list(map(tokenizer.convert_tokens_to_ids, test_tokens))
test_segment_ids = get_segment_idx(test_token_ids)

test_token_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), test_token_ids)
test_segement_ids = map(lambda tids: tids + [0] * (max_len - len(tids)), test_segment_ids)

bert_ = load_trained_model_from_checkpoint(
      "models/multi_cased_L-12_H-768_A-12/bert_config.json",
      checkpointName,
      training=True,
      trainable=True,
      seq_len=max_len)

print(bert_.summary())

def bert_mod(model):
	for layer in model.layers:
		layer.trainable = False 
	inputs = model.inputs
	dense = model.get_layer('Encoder-12-FeedForward-Norm').output
	#print(dense)
	#dense = np.array(dense)
	outputs = keras.layers.Dense(units=3, activation='softmax')(dense)

	model = keras.models.Model([model.inputs[0],model.inputs[1]], [outputs])
	model.compile(
  		optimizer='adam',
  		loss='sparse_categorical_crossentropy',
  		metrics=['accuracy'])

	return model

def batch_generator(ids,segement_ids,labels,batch_size):
	count = 0
	while True:
		for start in range(0, len(ids), batch_size):
			x = ids
			x1 = segement_ids
			y = labels
			x_batch = []
			x1_batch = []
			y_batch = []
			end = min(start + batch_size, len(x))
			ids_batch = x[start:end]
			segement_ids_batch = x1[start:end]
			label_batch = y[start:end]
			for id_,segement_id_,label in zip(ids_batch,segement_ids_batch,label_batch):
				count+=1
				#print(count,'llll')
				x_batch.append(id_)
				segement_batch.append(segement_id_)
				y_batch.append(label)
				if len(x_batch) == batch_size:
					b = np.array(x_batch)
					b1 = np.array(segement_batch)
					d = np.array(y_batch)
					#print(len(b))
					#print(len(d))
					yield [b,b],d 
def batch_gen(ids,labels,batch_size):
	count = 0
	while True:
		for start in range(0, len(ids), batch_size):
			x = ids
			y = labels
			x_batch = []
			y_batch = []
			end = min(start + batch_size, len(x))
			ids_batch = x[start:end]
			label_batch = y[start:end]
			for id_,segement_id_,label in zip(ids_batch,label_batch):
				count+=1
				#print(count,'llll')
				x_batch.append(id_)
				y_batch.append(label)
				if len(x_batch) == batch_size:
					b = np.array(x_batch)
					d = np.array(y_batch)
					#print(len(b))
					#print(len(d))
					yield [b,b],d 

model = bert_mod(bert_)

print(model.summary())



cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='bert_m.model',
	monitor='accuracy', verbose=1, save_best_only=True, mode='min')
train_x = list(train_token_ids)
test_x = list(test_token_ids)

batch_size = 2

train_gen = batch_gen(train_x,train_Y,batch_size)
test_gen = batch_gen(test_x,test_Y,batch_size)

model.fit(train_gen,epochs=100,verbose=1,steps_per_epoch=len(train_x)//batch_size)
'''