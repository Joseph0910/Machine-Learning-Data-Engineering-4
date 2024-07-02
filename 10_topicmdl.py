# Topic Modeling #

class TopicModeling:

	def __init__(self,document,components_n,iterations_n,word_mapping_doc,corpus_iterator,
				topics_n): # singular value decomp # 
		self.document = document #documents = ["doc1.txt", "doc2.txt", "doc3.txt"] 
		self.components_n = components_n
		self.iterations_n = iterations_n
		self.word_mapping_doc = word_mapping_doc
		self.corpus_iterator = corpus_iterator
		self.topics_n = topics_n

	def lsa(self):
		# raw documents to tf-idf matrix: 
		vectorizer = TfidfVectorizer(stop_words='english', 
		                             use_idf=True, 
		                             smooth_idf=True)

		# SVD to reduce dimensionality: 
		svd_model = TruncatedSVD(n_components=components_n,         // num dimensions
		                         algorithm='randomized',
		                         n_iter=iterations_n)

		# pipeline of tf-idf + SVD, fit to and applied to documents:
		svd_transformer = Pipeline([('tfidf', vectorizer), 
		                            ('svd', svd_model)])
		svd_matrix = svd_transformer.fit_transform(documents)

		# svd_matrix can later be used to compare documents, compare words, or compare queries with documents



	def plsa(self): # probabilistic latent semantic analysis #



	def lda(self): # Latent Dirichlet Allocation, bayesian approach # 

		# extract 100 LDA topics, updating once every 10,000
		lda = LdaModel(corpus=corpus_iterator, id2word=word_mapping_doc, num_topics=topics_n, update_every=1, chunksize=10000, passes=1)
		# use LDA model: transform new doc to bag-of-words, then apply lda
		doc_bow = doc2bow(document.split())
		doc_lda = lda[doc_bow]
		# doc_lda is vector of length num_topics representing weighted presence of each topic in the doc


	def lda_2vec(self):
		