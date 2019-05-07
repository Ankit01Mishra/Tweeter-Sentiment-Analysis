'''
This file contains the code of helper functions
which is used for cleaning the texts and preparing them 
for modelling 
'''

##necessary imports..
import codecs
import unidecode
import re
import spacy
import nltk
import collections
nlp = spacy.load('en')
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

##Basic Cleaning
def basic_clean(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text




'''
This is contraction mapping which is used to expand the english words.
'''

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" }



##It seems that we have problem with punctuations:--
def remove_punctuations(x):
	'''Slow not recommended ,better go with spacy cleaner
	'''
    x = str(x)
    for punct in '/-':
    	x = x.replace(punct,'')
    for punct in '&':
    	x = x.replace(punct,f'{punct}')
    
    for punct in "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–':
    	x = x.replace(punct,'')
    
    return x



def spacy_cleaner(text):
	'''
	text is the text column in the dataset which is to be cleaned

	It return the cleaned text..
	'''


    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)
    apostrophe_handled = re.sub("’", "'", decoded)
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected


##writting the helper function for build_dataset

def build_dataset(words, n_words):
    '''This function creates the required data format for passing in the RNN
	Arguments:--
	words:text corpus
	n_words:vocabulary size
	returns:
	returns the data in the format required to put in RNN(LSTM+GRU)
    '''
  
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words-1))
  
    ##initializing empty dictionary
    dictionary = dict()
    for word,_ in count:
    	dictionary[word] = len(dictionary)
    
    data = list()
    unk_count = 0
  	for word in words:
    	index = dictionary.get(word,0)
    	if index == 0:
      		unk_count += 1
    	data.append(index)
    count[0][1] = unk_count
  
  
    reversed_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
  
  	return data,count,dictionary,reversed_dictionary



def padder(text_padding = True,dictionary):
	'''
	This functions comes handy when the sequence size is less then 
	the requied length of the sequence
	'''
	GO = dictionary['GO']
	PAD = dictionary['PAD']
	EOS = dictionary['EOS']
	UNK = dictionary['UNK']

	return GO,PAD,EOS,UNK 

def str_idx(corpus, dic, maxlen, UNK=3):

	'''
	This function converts strings to numerical ids
	'''
  
    X = np.zeros((len(corpus),maxlen))
    for i in range(len(corpus)):
    	for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
      		try:
          		X[i,-1 - no]=dic[k]
      		except Exception as e:       
        		X[i,-1 - no]=UNK
    return X


def load_glove(path):
	'''This functon loads the pretrained glove word embedding 
	in the desired numpy format
	'''

	##creating the embedding_index
	embedding_index = {}
	f = open('glove.txt')
	for line in f:
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:],
		                    dtype = 'float32')
		embedding_index[word] = coefs
	  
	f.close()
	return embedding_index

def build_vocab(sentences,verbose = True):
	'''
	building vocan(dictionary) out of given word embeddings
	Arguments:
	sentences:--text data
	returns the vocab
	'''
    vocab = {}
    for sentence in tqdm(sentences,disable = (not verbose)):
    	for word in sentence:
        	try:
        		vocab[word] += 1
        except KeyError:
        	vocab[word] = 1
    return vocab


##checking the intersection of our vocab and the embeddings
import operator
def check_coverage(vocab,embedding_index):
	'''
	vocab:--vocab created using the embedding matrix
	embedding_index:--pretrained_word-embedding
	'''
    a = {}
    oov = {}
    k = 0
    i = 0
    for word in tqdm(vocab):
    	try:
      		a[word] = embedding_index[word]
      		k += vocab[word]
    	except:
      		oov[word] = vocab[word]
        	i += vocab[word]
        	pass
    print()  
    print("found embeddings for {} % of vocab".format((len(a)/len(vocab))*100))
    print("found embeddings for {} % of all texts".format((k/(k+i))*100))
  
    sorted_x = sorted(oov.items(),key = operator.itemgetter(1))[::-1]
    return sorted_x








