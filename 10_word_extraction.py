

from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
#nltk.download('wordnet') 
from nltk.stem.wordnet import WordNetLemmatizer
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import coo_matrix

# load the dataset
dataset = pandas.read_csv(file, delimiter = '\t')
dataset.head()

class WordExtraction():


	def __init__(self,dataset,word,corpus,coo_matrix,feature_names,sorted_items)
		self.dataset = dataset
		self.word = word
		self.corpus = corpus
		self.coo_matrix = coo_matrix
		self.feature_names = feature_names
		self.sorted_items = sorted_items

	def word_count(self):
		#Fetch wordcount for each abstract
		dataset['word_count'] = dataset['abstract1'].apply(lambda x: len(str(x).split(" ")))
		dataset[['abstract1','word_count']].head()

	def common_words(self):
		cmm_words = pandas.Series(' '.join(dataset['abstract1']).split()).value_counts()[:20]
		uncmm_words =  pandas.Series(' '.join(dataset ['abstract1']).split()).value_counts()[-20:]
		
		return cmm_words, uncmm_words

	def wrd_normalization(self):
		lem = WordNetLemmatizer()
		stem = PorterStemmer()

		print("stemming:",stem.stem(word))
		print("lemmatization:", lem.lemmatize(word, "v"))

	def remove_stopwords(self):
		##Creating a list of stop words and adding custom stopwords
		stop_words = set(stopwords.words("english"))
		##Creating a list of custom stopwords
		new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]
		stop_words = stop_words.union(new_words)

		corpus = []
		for i in range(0, len(dataset)):
		    #Remove punctuations
		    text = re.sub('[^a-zA-Z]', ' ', dataset['abstract1'][i])
		    
		    #Convert to lowercase
		    text = text.lower()
		    
		    #remove tags
		    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
		    
		    # remove special characters and digits
		    text=re.sub("(\\d|\\W)+"," ",text)
		    
		    ##Convert to list from string
		    text = text.split()
		    
		    ##Stemming
		    ps=PorterStemmer()
		    #Lemmatisation
		    lem = WordNetLemmatizer()
		    text = [lem.lemmatize(word) for word in text if not word in  
		            stop_words] 
		    text = " ".join(text)
		    corpus.append(text)

	def data_explore(self):
		wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stop_words,
                          max_words=100,
                          max_font_size=50, 
                          random_state=42
                         ).generate(str(corpus))
		print(wordcloud)
		fig = plt.figure(1)
		plt.imshow(wordcloud)
		plt.axis('off')
		plt.show()
		fig.savefig("word1.png", dpi=900)

	def word_count2(self):
		cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
		X=cv.fit_transform(corpus)
		return list(cv.vocabulary_.keys())[:10]

	def plot_unigram(self):

	    vec = CountVectorizer().fit(corpus)
	    bag_of_words = vec.transform(corpus)
	    sum_words = bag_of_words.sum(axis=0) 
	    words_freq = [(word, sum_words[0, idx]) for word, idx in      
	                   vec.vocabulary_.items()]
	    words_freq =sorted(words_freq, key = lambda x: x[1], 
	                       reverse=True)
	   
		#Convert most freq words to dataframe for plotting bar plot
		top_words = get_top_n_words(corpus, n=20)
		top_df = pandas.DataFrame(top_words)
		top_df.columns=["Word", "Freq"]

		#Barplot of most freq words
		sns.set(rc={'figure.figsize':(13,8)})
		g = sns.barplot(x="Word", y="Freq", data=top_df)
		g.set_xticklabels(g.get_xticklabels(), rotation=30)

		return g

	def bigram(self):
		#Most frequently occuring Bi-grams

	    vec1 = CountVectorizer(ngram_range=(2,2),  
	            max_features=2000).fit(corpus)
	    bag_of_words = vec1.transform(corpus)
	    sum_words = bag_of_words.sum(axis=0) 
	    words_freq = [(word, sum_words[0, idx]) for word, idx in     
	                  vec1.vocabulary_.items()]
	    words_freq =sorted(words_freq, key = lambda x: x[1], 
	                reverse=True)

		top2_words = get_top_n2_words(corpus, n=20)
		top2_df = pandas.DataFrame(top2_words)
		top2_df.columns=["Bi-gram", "Freq"]

		#Barplot of most freq Bi-grams
		sns.set(rc={'figure.figsize':(13,8)})
		h=sns.barplot(x="Bi-gram", y="Freq", data=top2_df)
		h.set_xticklabels(h.get_xticklabels(), rotation=45)

		return h


	def trigram(self):
	    vec1 = CountVectorizer(ngram_range=(3,3), 
	           max_features=2000).fit(corpus)
	    bag_of_words = vec1.transform(corpus)
	    sum_words = bag_of_words.sum(axis=0) 
	    words_freq = [(word, sum_words[0, idx]) for word, idx in     
	                  vec1.vocabulary_.items()]
	    words_freq =sorted(words_freq, key = lambda x: x[1], 
	                reverse=True)

		top3_words = get_top_n3_words(corpus, n=20)
		top3_df = pandas.DataFrame(top3_words)
		top3_df.columns=["Tri-gram", "Freq"]

		#Barplot of most freq Tri-grams
		sns.set(rc={'figure.figsize':(13,8)})
		j=sns.barplot(x="Tri-gram", y="Freq", data=top3_df)
		j.set_xticklabels(j.get_xticklabels(), rotation=45)	

		return j

	def conversion(self):
		tfidf_transformer=TfidfTransformer(smooth_idf=True,use_idf=True)
		tfidf_transformer.fit(X)
		# get feature names
		feature_names=cv.get_feature_names()
		 
		# fetch document for which keywords needs to be extracted
		doc=corpus[532]
		 
		#generate tf-idf for the given document
		tf_idf_vector=tfidf_transformer.transform(cv.transform([doc]))

	def sort(self):
	    tuples = zip(coo_matrix.col, coo_matrix.data)

	    return sorted(tuples, key=lambda x: (x[1], x[0]), reverse=True)
	 
	def extract_topn_from_vector(self):
    """get the feature names and tf-idf score of top n items"""
    
	    #use only topn items from vector
	    sorted_items = sorted_items[:10] # top 10
 
	    score_vals = []
	    feature_vals = []
    
    	# word index and corresponding tf-idf score
    	for idx, score in sorted_items:
	        #keep track of feature name and its corresponding score
	        score_vals.append(round(score, 3))
	        feature_vals.append(feature_names[idx])
 
	    #create a tuples of feature,score
	    #results = zip(feature_vals,score_vals)
	    results= {}
	    for idx in range(len(feature_vals)):
	        results[feature_vals[idx]]=score_vals[idx]
    
    	return results
		#sort the tf-idf vectors by descending order of scores
		sorted_items=sort_coo(tf_idf_vector.tocoo())
		#extract only the top n; n here is 10
		keywords=extract_topn_from_vector(feature_names,sorted_items,5)
 
		# now print the results
		print("\nAbstract:")
		print(doc)
		print("\nKeywords:")
		for k in keywords:
		    print(k,keywords[k])