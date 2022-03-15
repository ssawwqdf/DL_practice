import json

import pandas as pd
from flask import Flask, make_response, jsonify, request, render_template
import cx_Oracle
import sqlalchemy as sa
# from flask_cors import CORS, cross_origin
import requests
from bs4 import BeautifulSoup

from search import my_search_by_genres, my_search_by_review, my_search_by_meta

# ------------------------------------
#pip install flask_cors
# import flask_cors CORS, cross_origin
# CORS(app)
# CORS(app, resources={r'*': {'origins': '*'}})
# ------------------------------------



app = Flask(__name__, template_folder="template", static_folder="static")
# CORS(app)


@app.route('/')
def index():
    return render_template("index.html",)

@app.route('/search', methods=['POST', 'GET'])
def search():

    search_gubun = request.form.get('search_gubun')
    search_str = request.form.get('search_str')
    search_keyword = 'None'
    redirect_url = "result.html"

    res = []
    if search_gubun == 'genres': # 장르별
        res, search_keyword = my_search_by_genres(search_str, 0.97)

    elif search_gubun == 'story': # 스토리별
        res, search_keyword = my_search_by_review(search_str, 5)

    elif search_gubun == 'actor': # 배우/감독별
        res, search_keyword = my_search_by_meta(search_str, 5)  # Batman Forever

    else:
        redirect_url = "index.html"

    print('**'*30)
    print(search_gubun)
    print(search_str)
    print(res)
    print('**' * 30)

# result.html의 202번 라인 수정
# <p class="block-title" style="font-family: 'Black Han Sans', sans-serif;font-weight: 0;"><font size="10">{{SEARCH_KEYWORD}} : 추천 영화 BEST 5</font></p>
    return render_template(redirect_url, MY_INFO=res, SEARCH_KEYWORD = search_keyword)



# @app.route('/result')
# def result():
#     return render_template("result.html", )


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8088)