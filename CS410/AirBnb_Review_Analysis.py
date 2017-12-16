import pandas as pd
import numpy as np
import string
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize # function to split up our words
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder
from nltk import word_tokenize


#reviews=pd.read_csv("C:/reviews.csv")[0:2500]
reviews=pd.read_csv("C:/reviews.csv")

def get_language_likelihood(input_text):
    """Return a dictionary of languages and their likelihood of being the 
    natural language of the input text
    """
 
    input_text = input_text.lower()
    input_words = wordpunct_tokenize(input_text)
 
    language_likelihood = {}
    total_matches = 0
    for language in stopwords._fileids:
        language_likelihood[language] = len(set(input_words) &
                set(stopwords.words(language)))
 
    return language_likelihood

def get_language(input_text):
    """Return the most likely language of the given text
    """ 
    likelihoods = get_language_likelihood(input_text)
    return sorted(likelihoods, key=likelihoods.get, reverse=True)[0]

s=pd.DataFrame()
s['commentexist']=pd.notnull(reviews['comments'])
s['language']=[get_language(str(r)) for r in reviews['comments']]
s['listing']=reviews['listing_id']
s['comments']=reviews['comments']

s.to_csv("c:/s_langclassfied.csv", sep=',', encoding='utf-8')
s_filter=s[(s.commentexist==True)&(s.language=='english')]
sid = SentimentIntensityAnalyzer()
scored_reviews = pd.DataFrame()
pscores = [sid.polarity_scores(comment) for comment in s_filter['comments']]
scored_reviews['listing'] = [listing_id for listing_id in s_filter['listing']]
scored_reviews['review'] = [comments for comments in s_filter['comments']]
scored_reviews['positivity'] = [score['pos'] for score in pscores]

rev_group=scored_reviews.groupby('listing').apply(lambda df: (sum([positivity  for positivity  in df['positivity'].values]))/len([positivity  for positivity  in df['positivity'].values]))
rev_group_num=scored_reviews.groupby('listing').apply(lambda df: (len([positivity  for positivity  in df['positivity'].values])))
rev_group_f=pd.DataFrame()
rev_group_f['id']=rev_group.keys()
rev_group_f['sentiment_scores']=rev_group.values
rev_group_f['number of reviews']=rev_group_num.values

listing=pd.read_csv("c:/listings.csv")
list_few=listing[['id','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin','review_scores_communication','review_scores_location','review_scores_value','price','neighbourhood_cleansed']]
mergeddf=pd.merge(rev_group_f,list_few)
print(pd.merge(rev_group_f,list_few))
mergeddf.to_csv('C:/output-merged.csv', sep=',')


bigram_measures = BigramAssocMeasures()
#need a pandas data frame#exceeding 200 iam getting error..
#rev_few_colloc=rev_few
review_words =s_filter.groupby('listing').apply(lambda df: np.concatenate(np.array([word_tokenize(str(r)) for r in df['comments'].values])))
#ex = ['Hi', 'there', '.', '?', '!', ',']
ex=stopwords.words('english')+['Hi', 'there', '.', '?', '!', ',']
[w for w in ex if w not in string.punctuation]
review_words_f = review_words.map(lambda arr: np.array([w for w in arr if w not in string.punctuation]))

def reattach_contractions(wordlist):
    words = []
    for i, word in enumerate(wordlist):
        if word[0] == "'" or word == "n't":
            words[-1] = words[-1] + word
        else:
            words.append(word)
    return words

review_words_f = review_words_f.map(reattach_contractions)

def bigramify(words):
    finder = BigramCollocationFinder.from_words(words)
    finder.apply_freq_filter(3) 
    return finder.nbest(bigram_measures.pmi, 3)

review_bigrams = review_words_f.map(bigramify)
review_bigrams_pd=pd.DataFrame()
review_bigrams_pd['id']=review_bigrams.keys()
review_bigrams_pd['bigrams']=review_bigrams.values
review_bigrams_pd.to_csv('C:/output-bigrams.csv', sep=',')
print( type(review_bigrams))
print(review_bigrams)
