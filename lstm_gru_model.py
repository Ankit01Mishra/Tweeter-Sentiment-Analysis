##necessary imports...
import pandas as pd 
import matplotlib.pyplot as plt  
import numpy as np  
import tensorflow as tf  
from utils import * 

##Loading the dataset..
train = pd.read_csv("sentiment_analysis/train.csv")
test  = pd.read_csv("sentiment_analysis/test.csv")

##cleaning the texts..
train['clean_text'] = [spacy_cleaner(t) for t in train.tweet]
test['clean_text'] = [spacy_cleaner(t) for t in test.tweet]

##creating vocabulary using a word count based approcah

## TODO(later we'll include the trained word_embedding in place of build_dataset)
concat = ' '.join(train).split()
vocabulary_size = len(list(set(concat)))
data,count,dictionary,rev_dictionary = build_dataset(concat,vocabulary_size)
#print(vocabulary_size)

##hyperparameters
size_layer = 128
num_layers = 2
embedded_size = 128
dimension_output = 3
learning_rate = 0.01
maxlen = 50
batch_size = 128

##padding...
GO,PAD,EOS,UNK = padder(test_padding = True,dictionary = dictionary)


##defining the LSTM MODEL
class LSTM_GRUModel(object):
    def __init__(self,size_layer,num_layers,embedded_size,dict_size,dimension_output,learning_rate,isLSTM = False):
    '''it initializes the class with all the required variables'''
    
	    ##this function creates a cell for the LSTM.
	    def cells(reuse = False):
	        if isLSTM:
	        	return tf.nn.rnn_cell.LSTMCell(size_layer,initializer = tf.orthogonal_initializer(),reuse = reuse)
	        return tf.nn.rnn_cell.GRUCell(size_layer,initializer = tf.orthogonal_initializer(),reuse = reuse)
	    
	    self.X = tf.placeholder(tf.int32,[None,None])
	    self.Y = tf.placeholder(tf.float32,[None,dimension_output])
	    
	    encoder_embeddings = tf.Variable(tf.random_uniform([dict_size,embedded_size],-1,1))
	    encoder_embedded = tf.nn.embedding_lookup(encoder_embeddings,self.X)
	    
	    rnn_cells = tf.nn.rnn_cell.MultiRNNCell([cells() for _ in range(num_layers)])
	    
	    outputs,_ = tf.nn.dynamic_rnn(rnn_cells,encoder_embedded,dtype = tf.float32)
	    
	    W = tf.get_variable('w',shape = (size_layer,dimension_output),initializer = tf.orthogonal_initializer())
	    b = tf.get_variable('b',shape = (dimension_output),initializer = tf.zeros_initializer())
	    
	    self.logits = tf.matmul(outputs[:,-1],W)+b
	    
	    self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.logits,labels = self.Y))
	    self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)
	    
	    correct_pred = tf.equal(tf.argmax(self.logits,1),tf.argmax(self.Y,1))
	    self.accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

#Training
import os
tf.reset_default_graph()
sess = tf.InteractiveSession()
model = LSTM_GRUModel(size_layer,num_layers,embedded_size,vocabulary_size+4,dimension_output,learning_rate,isLSTM = True)
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)
checkpoint_dir = os.path.abspath(os.path.join('./', "checkpoints_rnn_lstm"))
checkpoint_prefix = os.path.join(checkpoint_dir, "model")


EARLY_STOPPING, CURRENT_CHECKPOINT, CURRENT_ACC, EPOCH = 5, 0, 0, 0
while True:
    lasttime = time.time()
    if CURRENT_CHECKPOINT == EARLY_STOPPING:
    	print('break epoch : %d\n'%(EPOCH))
    	break
    
    train_acc,train_loss,test_acc,test_loss = 0,0,0,0
  
    for i in range(0,(len(x_train)//batch_size)*batch_size,batch_size):
    
	    batch_x = str_idx(x_train[i:i+batch_size],dictionary,maxlen)
	    acc,loss,_ = sess.run([model.accuracy,model.cost,model.optimizer],
	                         feed_dict = { model.X:batch_x,model.Y : onehot_train[i:i+batch_size]})
	    
	    train_loss += loss
	    train_acc += acc
    
    for i in range(0,(len(x_test)//batch_size)*batch_size,batch_size):
	    batch_x = str_idx(x_test[i:i+batch_size],dictionary,maxlen)
	    acc,loss = sess.run([model.accuracy,model.cost],
	                       feed_dict = {model.X : batch_x,model.Y : onehot_test[i:i+batch_size]})
	    
	    test_loss += loss
	    test_acc += acc
    
    train_loss /= (len(x_train) // batch_size)
    train_acc /= (len(x_train) // batch_size)
    test_loss /= (len(x_test) // batch_size)
    test_acc /= (len(x_test) // batch_size)
  
    if test_acc > CURRENT_ACC:
	    print('epoch: %d,pass acc: %f,current_acc: %f'%(EPOCH,CURRENT_ACC,test_acc))
	    CURRENT_ACC = test_acc
	    CURRENT_CHECKPOINT = 0
    
    else:
    	CURRENT_CHECKPOINT += 1
    print('time taken: ',time.time()-lasttime)
    print('epoch: %d, training loss: %f, training acc: %f, valid loss: %f, valid acc: %f\n'%(EPOCH,train_loss,train_acc,test_acc,test_acc))
    EPOCH += 1



##Evaluation
logits = sess.run(model.logits,feed_dict = {model.X : str_idx(x_test,dictionary,maxlen)})
from sklearn.metrics import classification_report 
report = classification_report(y_test,np.argmax(logits,1))
print(report)