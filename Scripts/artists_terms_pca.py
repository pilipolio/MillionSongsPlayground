from millionsongs.io import db_loader, get_artists_from_csv
from millionsongs.indexes import lil_inverse_index

# Load artists, terms + indexes and artists_terms from raw metadata tables
loader=db_loader(r'D:\Data')
artists,terms,by_artist_terms,artists_terms_by_terms = loader.load()
artists_by_ids = dict((a[0],a) for a in artists)
artists_by_names = dict((a[2].lower(),a) for a in artists)

print '{} artists with {} unique ids and {} unique names'.format(len(artists), len(artists_by_ids), len(artists_by_names))
print '{} terms and {} term x artist_id tuples'.format(len(terms), len(artists_terms_by_terms))

# Build and fill boolean inverted indexes 
terms_artists_index = lil_inverse_index(terms,[a[0] for a in artists])
terms_artists_index.fill((t,id) for id,t in artists_terms_by_terms)

artists_terms_index = lil_inverse_index([a[0] for a in artists],terms)
artists_terms_index.fill((id,t) for id,t in by_artist_terms)

print '{}x{} boolean tf-idf matrix with {} non-null elements build from {} tuples'.format(terms_artists_index.inc_matrix.shape[0],terms_artists_index.inc_matrix.shape[1],terms_artists_index.inc_matrix.sum(),len(artists_terms_by_terms))

# Cumulating number of artists per term
by_terms_nartists = np.squeeze(np.array(terms_artists_index.inc_matrix.sum(axis=1)))
plt.hist(by_terms_nartists,bins=np.logspace(0,5,base=10,num=51,endpoint=True))
plt.xscale('log')
plt.xlabel('Terms count (n_artists = {}'.format(len(artists)))
plt.hist(100 * by_terms_nartists/len(artists),bins=np.logspace(0,2,base=10,num=21,endpoint=True))
plt.xscale('log')
plt.xlabel('Terms frequency')

# DEBUG
# Example : term 'classical' incidents to artists : 
terms_artists_index.boolean_query(['classical'])
[artists_by_ids[id][2] for id in terms_artists_index.boolean_query(['classical'])]
[artists_by_ids[id][2] for id in terms_artists_index.boolean_query(['french','classical','piano','chamber music'])]
[artists_by_ids[id][2] for id in terms_artists_index.boolean_query(['french','hip hop','rap','french rap','underground'])]

artists_terms_index.boolean_query([artists_by_names['wolfgang amadeus mozart'][0]])
artists_terms_index.boolean_query([artists_by_names['vitalic'][0],artists_by_names['laurent garnier'][0],artists_by_names['daft punk'][0]])

# import scipy's singular value decomposition
# See here for review of python implementations/wrappers of SVD
# http://jakevdp.github.com/blog/2012/12/19/sparse-svds-in-python/
from scipy.sparse.linalg import svds
# Transform sparse matrix from lil to csc for better performances, and calculate the 100 first singular values and vectors.
csc_m = terms_artists_index.inc_matrix.tocsc()
U_k,Sigma_k,V_k = svds(csc_m,k=100)

plt.plot(Sigma_k[::-1],label='boolean tf')

# inverse document frequency normalization to take into account that some terms are so frequent
# http://en.wikipedia.org/wiki/Tf%E2%80%93idf
# \mathrm{idf}(t, D) = \log \frac{|D|}{|\{d \in D: t \in d\}|} 
# \mathrm{tfidf}(t,d,D) = \mathrm{tf}(t,d) \times \mathrm{idf}(t, D)
idf = np.log(len(artists)/by_terms_nartists)
tf_idf = terms_artists_index.inc_matrix.tocsc()
# See http://stackoverflow.com/questions/12237954/multiplying-elements-in-a-sparse-array-with-rows-in-matrix for multiplying trick
tf_idf.data *= idf[tf_idf.indices]

# (see http://en.wikipedia.org/wiki/Latent_semantic_analysis and http://en.wikipedia.org/wiki/Latent_semantic_indexing)
# X_k = U_k \Sigma_k V_k^T
# Uk : (7643, k)
# Sigma_k = (k)
# V_k : (k, 44745)

U_k,Sigma_k,V_k = svds(tf_idf,k=100)
plt.plot(Sigma_k[::-1],label='tf_idf') 

## Project terms onto the 2d-subspace of vectors
terms_2d = np.dot(U_k[:,-2:],np.diag(Sigma_k[-2:]))
terms_2d = np.dot(U_k[:,-3:-1],np.diag(Sigma_k[-3:-1]))

# Compute terms norms the 2d space and find the 100 greatest
norms_terms_2d = np.array([numpy.linalg.norm(t) for t in terms_2d])
i_terms = np.where(norms_terms_2d>=mquantiles(norms_terms_2d,prob=1-100./len(terms))[0])[0]
plt.hist(norms_terms_2d,bins=np.logspace(0,3,base=10,num=31,endpoint=True))
plt.xscale('log')

plt.figure(figsize=(32,16))
# Display cloud of terms and 100 greatest on the 2d subspace
plt.scatter(terms_2d[:,0],terms_2d[:,1],marker='x',s=2.5,c='lightgrey')
for i in i_terms : plt.text(terms_2d[i,0],terms_2d[i,1],s=terms[i],ha='center',va='center')
plt.savefig('D:\PythonWorkspace\millionsongs\svd_tfidf_terms_2-3.png')
plt.close()

## Project artists onto the 2d-subspace of vectors
artists_1_2 = np.dot(np.diag(Sigma_k[-2:]), V_k[-2:,:])
artists_2_3 = np.dot(np.diag(Sigma_k[-3:-1]), V_k[-3:-1,:])
artists_2d = artists_2_3
artists_2d = np.dot(np.diag(Sigma_k[-2:]), V_k[-2:,:])
	
plt.figure(figsize=(32,16))
plt.scatter(artists_2d[0,:],artists_2d[1,:],marker='x',s=2.5,c='lightgrey')

## Plot a sample of we7 smart radio play-list
import millionsongs.io
artists_paths = [
	millionsongs.io.get_artists_from_csv(r'D:\Data\we7_path_1.csv'),
	millionsongs.io.get_artists_from_csv(r'D:\Data\we7_path_2.csv'),
	millionsongs.io.get_artists_from_csv(r'D:\Data\we7_path_3.csv')]

# Plot artists from playlists
plt.scatter(artists_2d[0,:],artists_2d[1,:],marker='x',s=2.5,c='lightgrey')
colors = ['blue','green','red']
for i,path in enumerate(artists_paths):
	 indexes = np.array([terms_artists_index.index_by_docs[artists_by_names[a][0]] for a in path if a in artists_by_names])
	 for a_i in indexes:
		 plt.text(x=artists_2d[0,a_i],y=artists_2d[1,a_i],s=artists[a_i][2],color=colors[i],ha='center',va='center')

plt.xlim([-4,5])
plt.ylim([-5,5])
plt.savefig('D:\PythonWorkspace\millionsongs\svd_tfidf_artists_2-3.png')
plt.close()

#http://pypi.python.org/pypi/gensim