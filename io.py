import sqlite3
from os import path
import codecs

"""
IO functions from sqlite databases found here (http://labrosa.ee.columbia.edu/millionsong/pages/getting-dataset)
"""
def get_artists_from_csv(csv_file):
	 with codecs.open(csv_file,encoding='utf-8') as f :
		 artists_path  = [unicode(l.replace('\r','').replace('\n','').split(';')[1].lower()) for l in f.readlines()]
	 return artists_path

def get_artists_from_db(track_db_file):
	 """ Return all artists attributes : id, name, ...
	 >>> len(get_artists_from_db(path.join('D:\Data','track_metadata.db')))
	 44745
	 """
	 conn = sqlite3.connect(track_db_file)
	 c = conn.cursor()
	 q = """
		SELECT artist_id,artist_mbid,artist_name FROM songs
		GROUP BY artist_id ORDER BY artist_id,artist_mbid,artist_name
		"""
	 res = c.execute(q)
	 return res.fetchall()

def get_artists_ids_from_db(artists_db_file):
	 """ Return all artists
	 >>> len(get_artists_ids_from_db(path.join('D:\Data','artist_term.db')))
	 44745
	 """
	 conn = sqlite3.connect(artists_db_file)
	 c = conn.cursor()
	 q = 'SELECT artist_id FROM artists'
	 res = c.execute(q)
	 return list(r[0] for r in res)
	 
def get_terms_from_db(terms_db_file):
	 """ Return all terms
	 >>> len(get_terms_from_db(path.join('D:\Data','artist_term.db')))
	 7643
	 """
	 conn = sqlite3.connect(terms_db_file)
	 c = conn.cursor()
	 q = 'SELECT term FROM terms ORDER by term'
	 res = c.execute(q)
	 return list(r[0] for r in res)

def get_artist_terms_by_artists_from_db(terms_db_file):
	 """ Return all artists_terms ordered by artists
	 >>> len(get_artist_terms_by_artists_from_db(path.join('D:\Data','artist_term.db')))
	 1109381
	 """
	 conn = sqlite3.connect(terms_db_file)
	 c = conn.cursor()
	 q = 'SELECT artist_id, term FROM artist_term ORDER by artist_id,term'
	 res = c.execute(q)
	 return res.fetchall()

def get_artist_terms_by_terms_from_db(terms_db_file):
	 """ Return all artists_terms ordered by terms
	 >>> len(get_artist_terms_by_terms_from_db(path.join('D:\Data','artist_term.db')))
	 1109381
	 """
	 conn = sqlite3.connect(terms_db_file)
	 c = conn.cursor()
	 q = 'SELECT artist_id, term FROM artist_term ORDER by term,artist_id'
	 res = c.execute(q)
	 return res.fetchall()

class db_loader:
	 """ DB loader to fetch artists name/ids, terms and artists to terms from sql lite db files
	 >>> l=db_loader(r'D:\Data')
	 >>> l.get_track_db_file()
	 'D:\\\\Data\\\\track_metadata.db'
	 >>> l.get_artists_terms_db_file()
	 'D:\\\\Data\\\\artist_term.db'
	 """
	 def __init__(self,db_directory):
		 self.db_directory = db_directory
	 
	 def get_track_db_file(self):
		 """ Returned tracks db file path
		 """
		 return path.join(self.db_directory,'track_metadata.db')

	 def get_artists_terms_db_file(self):
		 """ Returned artists/terms db file path
		 """
		 return path.join(self.db_directory,'artist_term.db')
	 
	 def load(self):
		 artists = get_artists_from_db(self.get_track_db_file())
		 terms = get_terms_from_db(self.get_artists_terms_db_file())
		 by_artist_terms = get_artist_terms_by_artists_from_db(self.get_artists_terms_db_file())
		 by_terms_artists = get_artist_terms_by_terms_from_db(self.get_artists_terms_db_file())
		 return (artists,terms,by_artist_terms,by_terms_artists)


if __name__ == "__main__":
    import doctest
    doctest.testmod()