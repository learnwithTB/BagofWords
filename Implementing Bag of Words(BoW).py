#!/usr/bin/env python
# coding: utf-8

# # Implementing Bag of Words(BoW)
# Conversion of Text into Vectorize form

# In[ ]:


import pandas as pd


# In[4]:


reviews=pd.read_excel('reviews.xlsx')


# In[5]:


reviews.head()


# In[6]:


reviews.shape


# In[8]:


import nltk
import re
from nltk.corpus import stopwords
### Dataset Preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(reviews)):
    print(i)
    title = re.sub('[^a-zA-Z]', ' ', reviews['Reviews'][i])
    title = title.lower()
    title = title.split()
    
    title = [ps.stem(word) for word in title if not word in stopwords.words('english')]
    title = ' '.join(title)
    corpus.append(title)


# In[9]:


corpus


# # Creating the BoW(Bag of words)

# In[25]:


from sklearn.feature_extraction.text import CountVectorizer


#Creating the bag of words
bow_article = CountVectorizer().fit(corpus)

count_tokens=bow_article.get_feature_names()

article_vect = bow_article.transform(corpus)
count_tokens


# In[26]:



df_count_vect=pd.DataFrame(data=article_vect.toarray(),columns=count_tokens)
df_count_vect


# In[ ]:




