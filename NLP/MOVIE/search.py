import pandas as pd
import numpy as np

# ---------------------------- sys ----------------------------

import re
import datetime as dt

import warnings

import os
import shutil # shutil.rmtree

# ---------------------------- NLP ----------------------------

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.metrics.pairwise import linear_kernel
# from ast import literal_eval # iterable(string) -> object


# ------------- tensorflow & keras -----------------
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Activation                 #-------------FC
# from keras.layers import Conv2D, MaxPooling2D,Flatten      #-------------CNN
# from keras.layers import LSTM                              #-------------RNN
# from keras.preprocessing.image import ImageDataGenerator   #-------------Augmentation
# from keras.preprocessing.image import array_to_img, img_to_array, load_img # ㄴ flow
#
#
# from tensorflow.keras.utils import to_categorical
# from keras.callbacks import EarlyStopping, ModelCheckpoint  #------------ callback

# ------------------- crawling ---------------------
import requests
from bs4 import BeautifulSoup

# ----------------- fixing seed --------------------
np.random.seed(1024)
# tf.random.set_seed(1024)

# ----------------- pre -setting --------------------
warnings.filterwarnings(action='ignore')

# ===================================================
# Data Load
# ===================================================
df = pd.read_csv('./dataset/movie_data.csv')

# ===================================================
# cosine simlarity for searching
# ===================================================

# -------- cosine_similarity
# view_tag
tfidf_view_tag = TfidfVectorizer(stop_words='english') #, ngram_range=(1,2), max_df=0.8, min_df=0.2)
res_view_tag =  tfidf_view_tag.fit_transform(df['view_tag'])

cos_sim_view_tag = cosine_similarity(res_view_tag)
cos_sim_view_tag_df = pd.DataFrame(cos_sim_view_tag)

# search4
df['search4'] = df['search4'].astype('str')
tfidf_search4 = TfidfVectorizer(stop_words='english')
res_search4 =  tfidf_search4.fit_transform(df['search4'])

cos_sim_search4 = cosine_similarity(res_search4)
cos_sim_search4_df = pd.DataFrame(cos_sim_view_tag)



# ===================================================
# Method for searching
# ===================================================

# ignore case & space & hyphen & colon
def search_engine(title):
    df_search = pd.DataFrame(df['title'])
    # df_search['']
    df_search['rm_blank'] = df_search['title'].apply(lambda x: re.sub('(\s)', '', x))
    df_search['rm_blank'] = df_search['title'].apply(lambda x: re.sub('(-)', '', x))
    df_search['rm_blank'] = df_search['title'].apply(lambda x: re.sub('(:)', '', x))

    title = re.sub('(\s)', '', title)

    try:
        title = df_search[df_search['rm_blank'].str.contains(title, case=False)]['title'].values[0]
    except:
        title = "Toy Story"

    return title


def my_search_by_genres(search_genres='Family', percent=0.95):
    m = df['vote_count'].quantile(0.95)
    # df5 = df[df['vote_count'] > m][['id', 'title', 'genres', 'vote_average', 'vote_count', 'year', 'wr', 'poster_path']]
    df5 = df[df['vote_count'] > m][['title','poster_path', 'overview', 'wr', 'genres']]
    df5 = df5.sort_values('wr', ascending=False)
    df5 = df5[df5['genres'].str.contains(search_genres, case = False)].head()

    res = []
    title_list = list(df5['title'])
    poster_list = list(df5['poster_path'])
    overview_list = list(df5['overview'])

    for i in range(5):
        res_list = [title_list[i], poster_list[i], overview_list[i]]
        res.append(res_list)

    return res, search_genres


def my_search_by_review(title="Toy Story", topn=5):
    # 공백, 대소문자, -, : 무시
    title = search_engine(title)

    # 인덱스 출력하기
    s = df['title']
    title_s = pd.Series(s.index, index=s.values)  # values <--> index
    idx = title_s[title]

    # 동명 영화 여러개인 경우 최신 영화
    if type(idx) == pd.Series:
        idx = idx[-1]

    idx_list = pd.Series(cos_sim_view_tag[idx].reshape(-1)).sort_values(ascending=False).index[1:topn + 1]  # 0번재는 본인. 1~10번째
    df5 = df.loc[idx_list, ['title','poster_path', 'overview']].head()
    # df5 = df.loc[idx_list, ['id', 'title', 'genres', 'vote_average', 'vote_count', 'year', 'wr', 'poster_path']].head()

    res = []
    title_list = list(df5['title'])
    poster_list = list(df5['poster_path'])
    overview_list = list(df5['overview'])

    for i in range(5):
        res_list = [title_list[i], poster_list[i], overview_list[i]]
        res.append(res_list)

    return res,title

def my_search_by_meta(title="Toy Story", topn=5):
    # 공백, 대소문자, -, : 무시
    title = search_engine(title)

    s = df['title']
    title_s = pd.Series(s.index, index=s.values)  # values <--> index
    idx = title_s[title]

    # 동명 영화 여러개인 경우 최신 영화
    if type(idx) == pd.Series:
        idx = idx[-1]

    idx_list = pd.Series(cos_sim_search4[idx].reshape(-1)).sort_values(ascending=False).index[1:topn + 1]  # 0번재는 본인. 1~10번째
    df5 = df.loc[idx_list, ['title','poster_path', 'overview']].head()

    res = []
    title_list = list(df5['title'])
    poster_list = list(df5['poster_path'])
    overview_list = list(df5['overview'])

    for i in range(5):
        res_list = [title_list[i], poster_list[i], overview_list[i]]
        res.append(res_list)

    return res, title
#
# def my_poster(title="Toy Story"):
#     title = title.replace(' ', '_')
#     url = f'https://en.wikipedia.org/wiki/{title}'
#     print(url)
#     res = requests.get(url)
#     soup = BeautifulSoup(res.text, 'html.parser')
#     # print(res)
#     img = soup.select_one('#mw-content-text > div.mw-parser-output > table.infobox.vevent > tbody > tr:nth-child(2) > td > a > img').get('src')
#     return img



# call
#
# res = my_search_wr_by_genres()
# print(res)
#
# res = my_search_cossim_by_review()
# print(res)
#
# res = my_search_cossim_by_search4()
# print(res)