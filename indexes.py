import numpy as np
from itertools import groupby
from scipy.sparse import lil_matrix

class lil_inverse_index:
	 """
	 Sparse list of list based words-documents inverse index
	 """
	 
	 def __init__(self,words,docs,index_by_words=None,index_by_docs=None):
		 """
		 Initialize an empty words-docs matrix
		 Index input words and docs if index with [] operator not provided
		 >>> i=lil_inverse_index(['a','b','c'],['doc1','doc2'])
		 >>> i.inc_matrix.shape
		 (3, 2)
		 >>> i=lil_inverse_index(['a','b','c'],['doc1','doc2'],{'a':0,'b':1,'c':2},{'doc1':0,'doc2':1})
		 >>> i.inc_matrix.shape
		 (3, 2)
		 """
		 self.words = words
		 self.docs = docs
		 if (index_by_words is None):
			 self.index_by_words = dict((w,i) for i,w in enumerate(words))
		 else:
			 self.index_by_words = index_by_words
		 if (index_by_docs is None):
			 self.index_by_docs = dict((d,i) for i,d in enumerate(docs))
		 else:
			 self.index_by_docs = index_by_docs
		 # I prefer to use float type for future normalization, sum or matrix multiplication
		 # which might be problematic with bool type.
		 self.inc_matrix = lil_matrix((len(words),len(docs)),dtype='float')
		 
	 def fill(self,word_doc_tuples):
		 """
		 Fill matrix row by row by going through all word_docs tuple
		 Take advantage of words order by grouping inserts
		 >>> i=lil_inverse_index(['a','b','c'],['doc1','doc2'])
		 >>> i.fill([('a','doc1')])
		 >>> i.inc_matrix.getrow(0).toarray()
		 array([[ 1.,  0.]])
		 """
		 for w,grouped_tuples in groupby(word_doc_tuples,lambda w_d : w_d[0]):
			 doc_indexes = [self.index_by_docs[d] for same_w,d in grouped_tuples]
			 self.inc_matrix[self.index_by_words[w],doc_indexes] = True

	 def boolean_query(self,words_query):
		 """
		 Simple AND boolean query returning a list of documents matching all words in query
		 >>> i=lil_inverse_index(['a','b','c'],['doc1','doc2'])
		 >>> i.fill([('a','doc1')])
		 >>> i.boolean_query(['a'])
		 ['doc1']
		 >>> i.fill([('a','doc2')])
		 >>> i.fill([('b','doc2')])
		 >>> i.fill([('c','doc2')])
		 >>> i.boolean_query(['a'])
		 ['doc1', 'doc2']
		 >>> i.boolean_query(['c'])
		 ['doc2']
		 """
		 words_indexes = np.array([self.index_by_words[w] for w in words_query if w in self.index_by_words])
		 return [self.docs[i] for i in np.where(self.inc_matrix[words_indexes,:].toarray().all(axis=0))[0]]
		 
if __name__ == "__main__":
    import doctest
    doctest.testmod()