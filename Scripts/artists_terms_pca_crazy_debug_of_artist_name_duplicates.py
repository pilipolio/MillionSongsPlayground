# Gui : connect to artist_term.db
from millionsongs.io import db_loader
from millionsongs.indexes import lil_inverse_index
import itertools

# Load artists, terms + indexes and artists_terms raw tables
loader=db_loader(r'D:\Data')
artists,terms,by_artist_terms,by_terms_artists = loader.load()
artists_by_ids = dict((a[0],a) for a in artists)
artists_by_names = dict((a[2].lower(),a) for a in artists)

print '{} artists with {} unique ids and {} unique names'.format(len(artists), len(artists_by_ids), len(artists_by_names))
print '{} terms and {} term x artist_id tuples'.format(len(terms), len(by_terms_artists))

# Build boolean inverted indexes 
terms_artists_index = lil_inverse_index(terms,[a[2].lower() for a in artists])
terms_artists_index.fill((id_t[1],artists_by_ids[id_t[0]][2].lower()) for id_t in by_terms_artists)

# debug
np.sum(terms_artists_index.inc_matrix.getrow(0).toarray())
debug_a = set(artists[int(i)][0] for i in terms_artists_index.inc_matrix.getrow(0).nonzero()[1])
len(debug_a)
check_a = set(a for a,t in by_terms_artists if t=='00s')
len(check_a)
np.unique(t for a,t in by_terms_artists[:3770])

len(check_a.difference(debug_a))
len(debug_a.difference(check_a))

check_a.difference(debug_a)
missing_a = [a for a,t in check_a if a not in debug_a]
too_much_a = [a for a in debug_a if a not in check_a]

artists_by_ids[u'ARC77IZ1187B9B91F6']
# id = u'ARC77IZ1187B9B91F6' and name = u'We Are Balboa' seem problematic (see artists_by_ids[u'ARC77IZ1187B9B91F6'])
# [(id,t) for id,t in by_terms_artists if id==u'ARC77IZ1187B9B91F6']
terms_artists_index = lil_inverse_index(terms,[a[2].lower() for a in artists])
terms_artists_index.fill((t,artists_by_ids[id][2].lower()) for id,t in by_terms_artists if id==u'ARC77IZ1187B9B91F6')
terms_artists_index.fill((t,artists_by_ids[id][2].lower()) for id,t in by_terms_artists if t==u'00s')
word_doc_tuples = [(t,artists_by_ids[id][2].lower()) for id,t in by_terms_artists if t==u'00s' and id==u'ARC77IZ1187B9B91F6']
word_doc_tuples = [(t,artists_by_ids[id][2].lower()) for id,t in by_terms_artists if t==u'00s']

terms_artists_index.fill(word_doc_tuples)
terms_artists_index.inc_matrix.getrow(0).nonzero()[1]


artist_in_word_doc_tuples = [artists_by_ids[id] for id,t in by_terms_artists if t==u'00s']
artist_name_in_word_doc_tuples = [artists_by_ids[id][2] for id,t in by_terms_artists if t==u'00s']
	 
# duplicates in by_terms_artists, for example u'woody van eyden' id = u'ARQELMK1187B9A49A3'
len(artist_name_in_word_doc_tuples)
len(set(artist_name_in_word_doc_tuples))

# same artist tuple (or at least name) in artists_by_ids!
len(artists_by_ids)
# Out[270]: 44745
len(set(a[2] for a in artists_by_ids.values()))
# Out[272]: 43538
#In [278]: len(artists)
#Out[278]: 44745
#In [279]: len(set(a[2] for a in artists))
#Out[279]: 43538
[a for a in artists if a[0] == u'ARQELMK1187B9A49A3']
[a for a in artists if a[2] == u'Woody van Eyden']
# Same artist's name with different id

# artists_by_names = dict((a[2].lower(),a) for a in artists)
# => did not throw exception whereas artists_by_names is smaller than artists and artists_by_ids

# fin debug	 
artists_terms_index = lil_inverse_index([a[2].lower() for a in artists],terms)
artists_terms_index.fill((artists_by_ids[id_t[0]][2].lower(),id_t[1]) for id_t in by_artist_terms)

terms_artists_index.inc_matrix.sum(axis=0).shape
terms_artists_index.inc_matrix.sum(axis=1).shape
artists_terms_index.inc_matrix.sum(axis=0).shape

artists_terms_index.inc_matrix.sum(axis=1)

by_terms_nartists = np.squeeze(np.array(terms_artists_index.inc_matrix.sum(axis=1)))
plt.hist(100 * by_terms_nartists/len(artists),bins=np.logspace(0,2,base=10,num=21,endpoint=True))

plt.hist(by_terms_nartists)
plt.xscale('log')

# try to plot cumulated counts...
#nartists_cum_counts = (np.cumsum(nartists_counts[0]),nartists_counts[1][:-1])

# DEBUG
# Example : term 'classical' incidents to artists : 
terms_artists_index.boolean_query(['classical'])
terms_artists_index.boolean_query(['french','classical','piano','chamber music'])
artists_terms_index.boolean_query(['Wolfgang Amadeus Mozart'.lower()])
artists_terms_index.boolean_query(['Vitalic'.lower(),'Laurent Garnier'.lower(),'Daft Punk'])
artists_terms_index.boolean_query(['Voices Of East Harlem'.lower()])

# See here for review of python implementations/wrappers of SVD
# http://jakevdp.github.com/blog/2012/12/19/sparse-svds-in-python/
from scipy.sparse.linalg import svds
m=terms_artists_index.inc_matrix.copy()
m.dtype=dtype('float')
U_k,Sigma_k,V_k = svds(m,k=10)
plt.plot(Sigma_k[::-1])

# inverse document frequency normalization to take into account that some terms are so frequent
# http://en.wikipedia.org/wiki/Tf%E2%80%93idf

from sklearn.preprocessing import normalize
m_normalized=terms_artists_index.inc_matrix.copy()
m_normalized.dtype=dtype('float')
m_normalized = normalize(m_normalized, norm='l1', axis=1)
# DEBUG : should be equal
1.0/len(m_normalized[0].nonzero()[1])
m_normalized[0][m_normalized[0].nonzero()]

U_k,Sigma_k,V_k = svds(m,k=10)
plt.plot(Sigma_k[::-1])
 

# SVD_k of the terms-artists matrix
# (see http://en.wikipedia.org/wiki/Latent_semantic_analysis and http://en.wikipedia.org/wiki/Latent_semantic_indexing)
# X_k = U_k \Sigma_k V_k^T
# Uk : (7643L, 2L)
# Sigma_k = (2L)
# V_k : (2L, 44745L)

## Display artists k-subspace
artists_2d = np.dot(np.diag(Sigma_k), V_k)
plt.scatter(artists_2d[0,:],artists_2d[1,:],marker='x',s=5,c='grey')

# Load a given path of artists
artists_path_file = r'D:\Data\we7_path_3.csv'
import codecs
with codecs.open(artists_path_file,encoding='utf-8') as f :
	 artists_path  = [unicode(l.replace('\r','').replace('\n','').split(';')[1].lower()) for l in f.readlines()]

artists_path = [artists_by_names[a] for a in artists_path if artists_by_names.has_key(a)]
artists_path_indexes = np.array([artists_terms_index.index_by_words[a[2].lower()] for a in artists_path])

## Plot a given we7 radio play-list
plt.scatter(artists_2d[0,artists_path_indexes],artists_2d[1,artists_path_indexes],marker='o',c='blue',s=10)
for i,a in enumerate(artists_path):
	 plt.text(x=artists_2d[0,artists_path_indexes[i]],y=artists_2d[1,artists_path_indexes[i]],s=a[2],color='green')

# Distribution of each terms norm in the 2d space
norms_terms_2d = np.array([numpy.linalg.norm(t) for t in terms_2d])
plt.hist(norms_terms_2d,bins=np.logspace(0,3,base=10,num=31,endpoint=True))
plt.xscale('log')

## Display terms k-subspace
terms_2d = np.dot(U_k,np.diag(Sigma_k))
plt.scatter(terms_2d[:,0],terms_2d[:,1],marker='x',s=5,c='grey')

i_terms = np.where(norms_terms_2d>=25)[0]
for i in i_terms:
	 plt.text(terms_2d[i,0],terms_2d[i,1],s=terms[i])
plt.ylim([0,160])
plt.text(terms_2d[terms.index('rock'),0],terms_2d[terms.index('rock'),1],s='rock')
plt.text(terms_2d[terms.index('pop'),0],terms_2d[terms.index('pop'),1],s='pop')
plt.text(terms_2d[terms.index('classical'),0],terms_2d[terms.index('classical'),1],s='classical')
plt.text(terms_2d[terms.index('hip hop'),0],terms_2d[terms.index('hip hop'),1],s='hip hop')
plt.text(terms_2d[terms.index('jazz'),0],terms_2d[terms.index('jazz'),1],s='jazz')

