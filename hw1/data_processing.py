import numpy as np
import codecs
import json
from gensim.models import KeyedVectors

UNK = 'UNK'
PAD = '<p>'

def createDataset(path, word_mapping=None, label_mapping=None, update_mapping=True):
	with open(path, 'rb') as f:
		s = f.read()
	print(s[:10])
	s = ''.join([chr(s[i]) for i in range(len(s)) if int.from_bytes(s[i:i+1], byteorder='big') < 128])
	data = s.strip().split('\n')	
	data = np.array([x.split('|||') for x in data])	
	labels, sents = data[:,0], data[:,1]
	labels = np.array([x for x in labels])
	
	max_len = max([len(x.split()) for x in sents])

	if word_mapping is None:
		word_mapping = {PAD:0, UNK:1}

	if label_mapping is None:
		label_mapping = {UNK:0}

	labels = w2i(labels, label_mapping, update_mapping)
	sents = [w2i(s.split(), word_mapping, update_mapping)[:max_len] for s in sents]
	sents = [x + [word_mapping[PAD]]*(max_len - len(x)) for x in sents]
	return labels, sents, (word_mapping, label_mapping)		

def w2i(wseq, mapping, update_mapping):
	if update_mapping:
		iseq = [mapping.setdefault(w, len(mapping.keys())) for w in wseq]
	else:
		iseq = [mapping.setdefault(w, mapping[UNK]) for w in wseq]
	return iseq

def getEmbeddings(vocab, mapping):
	embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), 300))
	word2vec = KeyedVectors.load_word2vec_format('/data/mshah1/GoogleNews-vectors-negative300.bin', binary=True)	
	for w in vocab:
		if w in word2vec:
			embeddings[mapping[w]] = word2vec[w]
	return embeddings

if __name__ == '__main__':
	y_train, x_train, (word_mapping, label_mapping) = createDataset('topicclass/topicclass_train.txt')
	json.dump(word_mapping, open('topicclass/word_mapping.json','w'))
	json.dump(label_mapping, open('topicclass/label_mapping.json','w'))
	np.savez('topicclass/train.npz', x=x_train, y=y_train)

	# y_valid, x_valid,_ = createDataset('topicclass/topicclass_valid.txt', word_mapping=word_mapping, label_mapping=label_mapping, update_mapping=False)
	# np.savez('topicclass/valid.npz', x=x_valid, y=y_valid)

	# y_test, x_test,_ = createDataset('topicclass/topicclass_test.txt', word_mapping=word_mapping, label_mapping=label_mapping, update_mapping=False)
	# np.savez('topicclass/test.npz', x=x_test, y=y_test)

	# with open('topicclass/word_mapping.json','r') as f:
	# 	word_mapping = json.load(f)
	# embed = getEmbeddings(word_mapping.keys(), word_mapping)
	# np.save('w2v.npy', embed)