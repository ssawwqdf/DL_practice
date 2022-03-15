import pandas as pd
import numpy as np

# ---------------------------- sys ----------------------------

import datetime as dt

import warnings


import os
import shutil # shutil.rmtree

#---------------- visualization (edcoding, grid) ---------------

import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------- NLP ----------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval # iterable(string) -> object


# ------------- tensorflow & keras -----------------
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation                 #-------------FC
from keras.layers import Conv2D, MaxPooling2D,Flatten      #-------------CNN
from keras.layers import LSTM                              #-------------RNN
from keras.preprocessing.image import ImageDataGenerator   #-------------Augmentation
from keras.preprocessing.image import array_to_img, img_to_array, load_img # ㄴ flow



from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint  #------------ callback

# --------------------- etc. -----------------------
from PIL import Image

# ----------------- fixing seed --------------------
np.random.seed(1024)
tf.random.set_seed(1024)

# ----------------- pre -setting --------------------
warnings.filterwarnings(action='ignore')

# ===================================================
# DataFrame for searching
# ===================================================

# -------- Data load
mdf = pd.read_csv("./dataset/movies_metadata_2.csv")
ldf = pd.read_csv('./dataset/links_small.csv')
cdf = pd.read_csv("./dataset/credits.csv")
kdf = pd.read_csv("./dataset/keywords.csv")



# -------- mdf pre-Processing
idx= mdf[mdf['id'].str.len()>6].index
mdf = mdf.drop(idx, axis=0)

mdf['id']=mdf['id'].astype('int')

# -------- ldf pre-Processing
ldf = ldf.dropna()

ldf['tmdbId'] = ldf['tmdbId'].astype('int')



# -------- join
mldf = pd.merge(mdf, ldf, left_on='id', right_on='tmdbId')
temp = pd.merge(mldf, cdf, left_on='id', right_on='id', how = 'inner')
df = pd.merge(temp, kdf, left_on='id', right_on='id', how = 'inner')
# df = df.reset_index(drop = True)



# -------- Derived variables : view_tag
mdf['tagline'] = mdf['tagline'].fillna('')
mdf['overview'] = mdf['overview'].fillna('')
mdf['view_tag'] = mdf['overview'] + mdf['tagline']

idx = mdf[mdf['view_tag']==''].index
mdf = mdf.drop(idx, axis = 0)


# -------- Derived variables : actor, director, key, genres
df['cast'] = df['cast'].apply(literal_eval)            # actor
df['crew'] = df['crew'].apply(literal_eval)            # director
df['keywords'] = df['keywords'].apply(literal_eval)    # main keyword
df['genres'] = df['genres'].apply(literal_eval)        # genres


def get_director(s):
    for dict in s:
        if dict['job'] == 'Director':
            dict['name'] = dict['name'].replace(' ', '')
            return [dict['name'].lower()]  # list로 return
        # return np.nan
def get_cast(s):
    cast_list= []
    for dict in s:
        dict['name'] = dict['name'].replace(' ', '')
        cast_list.append(dict['name'].lower())
    return cast_list[:3]


df['director'] = df['crew'].apply(get_director)
df['actor'] = df['cast'].apply(get_cast)
df['key'] = df['keywords'].apply(get_cast)
# df['genres'] = df['genres']

df['search4'] = df['director'] + df['actor'] + df['key'] + df['genres']

# -------- cosine_similarity
tfidf_view_tag = TfidfVectorizer(stop_words='english') #, ngram_range=(1,2), max_df=0.8, min_df=0.2)
res_view_tag =  tfidf.fit_transform(mldf['view_tag'])

cos_sim_view_tag = cosine_similarity(res_view_tag)


df['search4'] = df['search4'].astype('str')
tfidf_search4 = TfidfVectorizer(stop_words='english')
res_search4 =  tfidf.fit_transform(df['search4'])

cos_sim = cosine_similarity(res_search4)


# ===================================================
# Method for searching
# ===================================================

C = mdf['vote_average'].mean()
m = mdf['vote_count'].quantile(0.95)
def my_calc_wr(mdf):
    R = mdf['vote_average']
    v = mdf['vote_count']
    WR = (v / (v+m) * R + (m/(v+m)*C))
    return WR

def my_search_wr_by_genres(search_genres='Family', percent=0.95):
    C = mdf['vote_average'].mean()
    m = mdf['vote_count'].quantile(percent)
    mdf['wr'] = mdf.apply(my_calc_wr, axis=1)

    df5 = mdf[mdf['vote_count'] > m][['id', 'title', 'genres', 'vote_average', 'vote_count', 'year', 'wr']]
    df5 = df5.sort_values('wr', ascending=False)

    return df5[df5['genres'].str.contains(search_genres)].head()


def my_search_cossim_by_review(title="Toy Story", topn=10):
    # 인덱스 출력하기 다른 ver
    s = mldf['title']
    title_s = pd.Series(s.index, index=s.values)  # values <--> index
    idx = title_s[title]

    idx_list = pd.Series(cos_sim[idx].reshape(-1)).sort_values(ascending=False).index[1:topn + 1]  # 0번재는 본인. 1~10번째

    title_list = mldf.loc[idx_list, 'title'].values

    return title_list

def my_search_cossim_by_search4(title="Toy Story", topn=10):
    s = df['title']
    title_s = pd.Series(s.index, index=s.values)  # values <--> index
    idx = title_s[title]

    idx_list = pd.Series(cos_sim[idx].reshape(-1)).sort_values(ascending=False).index[1:topn + 1]  # 0번재는 본인. 1~10번째

    title_list = df.loc[idx_list, 'title'].values

    return title_list

def search_engine(title):
    df_search = df['title']
    df_search['lower'] = df['title'].str.lower()

    df_search['title']
