##necessary imports..
import tensorflow as tf 
from utils import spacy_cleaner
from utils import load_glove
from utils import build_vocab
from utils import check_coverage
import pandas as pd 
import nunpy as np  
import matplotlib.pyplot as plt  

##loading the dataset..
train = pd.read_csv("sentiment_analysis/train.csv")
test  = pd.read_csv("sentiment_analysis/test.csv")

##Loading the glove vectors...
embedding_index = load_glove('sentiment_analysis/glove.6B.100d.txt')

##building vocab
train['clean_text'] = [spacy_cleaner(t) for t in train.tweet]
sentences = train['clean_text'].map(lambda z: z.split())

vocab_step1 = build_vocab(sentences)

##checking the coverage..
oov = check_coverage(vocab_step1,embedding_index)

##Inspection...
print(oov[:20])
##this shows top-20 out of vocabulary words which are to be modified..
##Modification is done to make full use of word_embeddings

##This process is repeated iteratevely until coverage reaches more then 90%..

##lower..
##Building the sentences out of the questions:--
df['clean_text'] = df['clean_text'].map(lambda z : z.lower())
sentences = df['clean_text'].map(lambda z: z.split())

##creating the vocab
vocab = build_vocab(sentences)
##checking the coverage
oov  = check_coverage(vocab,embedding_index)

oov[:10]

correction = {'instagood':'very best photo','iphonex':'iphone','photooftheday':'photo of the day',
             'selfie':'self portrait type image','iphoneasia':'iphone asia','iphoneonly':'iphone only',
             'itune':'music application','picoftheday':'photo of the day','instamood':'mood of taking picture'}

def clean_contractions(text, mapping):
    text = ' '.join([mapping[t] if t in mapping else t for t in text.split(" ")])
    return text

df['clean_text'] = df['clean_text'].apply(lambda x: clean_contractions(x, correction))

sentences = df['clean_text'].map(lambda z: z.split())

##getting the coverage
##creating the vocab
vocab = build_vocab(sentences)
##checking the coverage
oov  = check_coverage(vocab,embedding_index)

oov[:10]

correction2 = {'iphonesia':'iphone asia','retweet':'tweet again','followme':'follow me','iphonea':'iphone',
              'fuckyou':'abuse','galaxys':'galaxy','instapic':'social media picture','christma':'christmas'}
df['clean_text'] = df['clean_text'].apply(lambda x: clean_contractions(x, correction2))

sentences = df['clean_text'].map(lambda z: z.split())

##getting the coverage
##creating the vocab
vocab = build_vocab(sentences)
##checking the coverage
oov  = check_coverage(vocab,embedding_index)

oov[:10]

correction3 = {'instadaily':'social media daily','followback':'follow back','macbookpro':'laptop','iphoneplus':'iphone',
               'hateapple':'hate company','newphone':'new phone','sonyphoto':'sony photo'}

df['clean_text'] = df['clean_text'].apply(lambda x: clean_contractions(x, correction3))

sentences = df['clean_text'].map(lambda z: z.split())

##getting the coverage
##creating the vocab
vocab = build_vocab(sentences)
##checking the coverage
oov  = check_coverage(vocab,embedding_index)


##-------------------------------------------------------------
##-------------------------------------------------------------
##                         DEEP LEARNING MODELLING...
##-------------------------------------------------------------
##-------------------------------------------------------------
from keras.preprocessing.text import Tokenizer
max_words = 100000
##on train
tokenizer_train = Tokenizer(num_words = max_words,lower = True,oov_token = 'UNK')
tokenizer_train.fit_on_texts(train['clean_text'])
word_ids = tokenizer_train.word_index
print(word_ids)

##on test
tokenizer_test = Tokenizer(num_words = max_words,lower = True,oov_token = 'UNK')
tokenizer_test.fit_on_texts(test['clean_text'].values)
word_ids_test = tokenizer_test.word_index

##converting the training data to the sequence of word indices:
seq_train = tokenizer_train.texts_to_sequences(train['clean_text'])
seq_test  = tokenizer_test.texts_to_sequences(test['clean_tweet'])

##padding the semtences so as to make them equal sized.
from keras.preprocessing.sequence import pad_sequences
max_sequence_length = 30
text_data = pad_sequences(seq_train,max_sequence_length)
text_data_test = pad_sequences(seq_test,max_sequence_length)

##creatig the embedding matrix

embedding_dim = 100

embedding_matrix = np.zeros(shape = (len(word_ids)+1,embedding_dim))

for word,i in word_ids.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        
        # words not found will be all 0s
        embedding_matrix[i] = embedding_vector
    
embedding_matrix_test = np.zeros(shape = (len(word_ids_test)+1,embedding_dim))

for word,i in word_ids_test.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # words not found will be all 0s
        embedding_matrix_test[i] = embedding_vector

##creating the network and training
from keras.layers import Embedding
embedding_layer = tf.keras.layers.Embedding(input_dim = len(word_ids)+1,
                           output_dim = embedding_dim,
                           weights = [embedding_matrix],
                           input_length = max_sequence_length,
                           trainable = True)

## Model definition
from keras.layers import Input,Dense,concatenate,Activation
input_data = tf.keras.layers.Input(shape = (max_sequence_length,),dtype = 'int32')
encoder = (embedding_layer)(input_data)

bigram_branch = tf.keras.layers.Conv1D(filters = 100,
                                  kernel_size = 2,
                                  padding = 'valid',
                                  activation = 'relu',
                                  strides = 1)(encoder)
bigram_branch = tf.keras.layers.GlobalAveragePooling1D()(bigram_branch)

##for kernel_size 3
trigram_branch = tf.keras.layers.Conv1D(filters = 100,
                                  kernel_size = 3,
                                  padding = 'valid',
                                  activation = 'relu',
                                  strides = 1)(encoder)
trigram_branch = tf.keras.layers.GlobalAveragePooling1D()(trigram_branch)

##for kernel_size 4
fourgram_branch = tf.keras.layers.Conv1D(filters = 100,
                                  kernel_size = 4,
                                  padding = 'valid',
                                  activation = 'relu',
                                  strides = 1)(encoder)
fourgram_branch = tf.keras.layers.GlobalAveragePooling1D()(fourgram_branch)

##Merging the layers
merged = tf.keras.layers.concatenate([bigram_branch,trigram_branch],axis = 1)

##calling the dense layer
merged = tf.keras.layers.Dense(256,activation = 'relu')(merged)
merged = tf.keras.layers.Dropout(0.4)(merged)
merged = tf.keras.layers.Dense(1)(merged)

##output
output = tf.keras.layers.Activation('sigmoid')(merged)

##calling Model
model = tf.keras.Model(inputs=[input_data], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()

##checkpointing directory
fd = 'weights-{epoch:02d}-{loss:.4f}.hdf5'
cp = tf.keras.callbacks.ModelCheckpoint(fd,monitor = 'val_acc',verbose = 1,save_best_only = True,mode = 'min')
checkpoint_list = [cp]

y_train = df['label'].values
y_train

##Hyperparameters
NB_EPOCHS = 10
VALIDATION_SPLIT = 0.09
BATCH_SIZE = 128

##Fitting the model
history = model.fit(text_data,
                    y_train,
                    epochs=NB_EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=checkpoint_list)

model.load_weights('weights-01-0.0590.hdf5')
loss, accuracy = model.evaluate(text_data_test, y_test, verbose=0)
print('loss = {0:.4f}, accuracy = {1:.4f}'.format(loss, accuracy))

prediction = model.predict(text_data_test)
y_pred = (prediction > 0.5)
from sklearn.metrics import f1_score, confusion_matrix
print('F1-score: {0}'.format(f1_score(y_pred, y_test)))
print('Confusion matrix:')
confusion_matrix(y_pred, y_test)
