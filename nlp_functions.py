# CLEANING DATA FOR NLP MACHINE LEARNING.

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Import a csv file with a text column, that is going to be used as a regressor for a ML model
# train_tab = pd.read_csv("labeledTrainData.tsv", header=0, delimiter="\t", quoting=3)
train =pd.read_csv("train_data.csv")
print(train)

# nltk.download() # Download "Models" / "punkt"
# nltk.download('stopwords')
nltk.download('wordnet') # If error: nltk.download() / 'wordnet' / chek solution: https://stackoverflow.com/questions/27750608/error-installing-nltk-supporting-packages-nltk-download

# The next function takes a raw text and transorm it to a clan text: 
# only words, lowlettered, common english words removed.

def text_to_words( raw_text ):
    # 1. Remove punctuation, tags, markups, numbers and stop words
    beautiful_text = BeautifulSoup(raw_text).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", beautiful_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #  Lemmatize
    lemma =[]
    lemmatizer = WordNetLemmatizer()
    for word in meaningful_words:
        lemma_word = lemmatizer.lemmatize(word, pos = "v")
        lemma.append(lemma_word)
    
    # 6. Join the words back into one string separated by space, and return the result.
    return( " ".join( lemma ))
    
# press enter after executing the function

clean_text = text_to_words( train["text"][0] )
print(clean_text)


# With this for loop you can pass the function to all rows in the data and clean them

num_data_rows = train["text"].size

clean_train_rows = []

for i in range(0, num_data_rows):
                  if ( (i+1)%1000 == 0 ):
                          print( "Review %d of %d\n" % (i +1, num_data_rows)  ) #print status of the loop
                  clean_train_rows.append( text_to_words( train["text"][i] ) ) 
# press enter after executing the command

# as you can see, the loop generates a 'clean_train_review' with the clean text ready for machine learning processing.
print(clean_train_rows[0])
print(clean_train_rows[1])
print(clean_train_rows[2])
print(clean_train_rows[3])
###################################################################################

# Functions for descriptive analysis:
# Make a word cloud for a natural language row of the 'clean_train_rows' list

from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Append the clean text to de data frame and separate in good and bad results
train["text_clean"] = clean_train_rows

positive_result = train[train["result"] == 1]
negative_result = train[train["result"] == 0]

all_positive_clean = " ".join( positive_result["text_clean"] )
all_negative_clean = " ".join( negative_result["text_clean"] )

# Print word cloud for good results
wordcloud1 = WordCloud()
wordcloud1.generate(all_positive_clean)
wordcloud1.to_image()

plt.imshow(wordcloud1)
plt.axis("off")
plt.show()
wordcloud1.to_file("wordcloud1.png")
plt.clf()

# Print word cloud for bad results
wordcloud0 = WordCloud()
wordcloud0.generate(all_negative_clean)
wordcloud0.to_image()

plt.imshow(wordcloud0)
plt.axis("off")
plt.show()
wordcloud0.to_file("wordcloud2.png")
plt.clf()

######################################################################################

# Convert the letters list to a numeric form, ready to use in ML

from sklearn.feature_extraction.text import CountVectorizer

# list converted to numeric features
vectorizer = CountVectorizer()
matrix = vectorizer.fit_transform(clean_train_rows)
matrix.shape
print(matrix)

# table visualization for understanding
matrix_df = pd.DataFrame(matrix.toarray())
matrix_df.columns = vectorizer.get_feature_names()
matrix_df.shape
print(matrix_df)

