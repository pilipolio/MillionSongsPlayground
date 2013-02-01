# Gui : connect to artist_term.db
from millionsongs.io import db_loader
from millionsongs.indexes import lil_inverse_index
import itertools

# Load artists, terms + indexes and artists_terms raw tables
loader=db_loader(r'D:\Data')
artists,terms,by_artist_terms,by_terms_artists = loader.load()
artists_by_ids = dict((a[0],a) for i,a in enumerate(artists))
artists_by_names = dict((a[2].lower(),a) for a in artists)

# Build boolean inverted indexes 
terms_artists_index = lil_inverse_index(terms,[a[2].lower() for a in artists])
terms_artists_index.fill((id_t[1],artists_by_ids[id_t[0]][2].lower()) for id_t in by_terms_artists)

artists_terms_index = lil_inverse_index([a[2].lower() for a in artists],terms)
artists_terms_index.fill((artists_by_ids[id_t[0]][2].lower(),id_t[1]) for id_t in by_artist_terms)

# Distribution of number of terms per artists
by_artists_nterms= np.array(map(
	lambda (a,ts) : (a,len(list(ts))),
	itertools.groupby(by_artist_terms,lambda a_t : a_t[0])),
	dtype=[('artist', '|S10'), ('n_terms', '<i4')])

# Distribution of number of artists per terms
by_terms_nartists= np.array(map(
	lambda (t,ars) : (t,len(list(ars))),
	itertools.groupby(by_terms_artists,lambda a_t : a_t[1])),
	dtype=[('term', '|S10'), ('n_artists', '<i4')])

np.sort(by_terms_nartists,order='n_artists')[-20:-1]

#plt.hist([100 * np.float(t_n[1])/len(artists) for t_n in by_terms_nartists],bins=np.linspace(0,100,num=20))
nartists_counts = plt.hist([100 * np.float(t_n[1])/len(artists) for t_n in by_terms_nartists],bins=np.logspace(0,2,base=10,num=21,endpoint=True))
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

from scipy.sparse.linalg import svds
m=terms_artists_index.inc_matrix.copy()
m.dtype=dtype('float')
U_k,Sigma_k,V_k = svds(m,k=2)

# SVD_k of the terms-artists matrix
#  X_k = U_k \Sigma_k V_k^T
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

## Display artists k-subspace
terms_2d = np.dot(np.diag(Sigma_k), V_k)
plt.scatter(artists_2d[0,:],artists_2d[1,:],marker='x',s=5,c='grey')

from sklearn.decomposition import RandomizedPCA

# pca of artist on terms
pca = RandomizedPCA(n_components = 2)
pca.fit(terms_by_artists)
artists_2d = terms_by_artists * pca.components_.transpose()
plt.scatter(artists_2d[:,0],artists_2d[:,1],marker='x',s=5,c='grey')

plt.plot([0,pca.components_[0,terms.index('rock')]],[0,pca.components_[1,terms.index('rock')]],label='rock')
plt.plot([0,pca.components_[0,terms.index('pop')]],[0,pca.components_[1,terms.index('pop')]],label='pop')
plt.plot([0,pca.components_[0,terms.index('classic')]],[0,pca.components_[1,terms.index('classic')]],label='classic')
plt.plot([0,pca.components_[0,terms.index('hip hop')]],[0,pca.components_[1,terms.index('hip hop')]],label='hip hop')
plt.plot([0,pca.components_[0,terms.index('jazz')]],[0,pca.components_[1,terms.index('jazz')]],label='jazz')

# pca of terms on artists
pca = RandomizedPCA(n_components = 2)
pca.fit(terms_by_artists.transpose())
terms_2d = terms_by_artists.transpose() * pca.components_.transpose()
plt.scatter(terms_2d[:,0],terms_2d[:,1])
for i in range(len(terms)): plt.text(terms_2d[i,0],terms_2d[i,1],terms[i])