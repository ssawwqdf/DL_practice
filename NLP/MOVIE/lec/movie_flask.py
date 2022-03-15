import json

import pandas as pd
from flask import Flask, make_response, jsonify, request, render_template
import cx_Oracle
import sqlalchemy as sa
# from flask_cors import CORS, cross_origin
import requests
from bs4 import BeautifulSoup

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

    # search_gubun = request.args.get('search_gubun')
    # search_str = request.args.get('search_str')
    redirect_url = "result.html"

    # if search_gubun == '장르별':
    #     resdf = my_search_by_genres('Fantasy', 0.97)
    #     print(res)  #제목,포스터,리뷰
    #
    # elif search_gubun == '스토리별':
    #     res = my_search_by_review('TOY', 5)
    #     print(res)   #제목,포스터,리뷰
    # elif search_gubun == '배우/감독별':
    #     res = my_search_by_meta('toy', 5)  # Batman Forever
    #     print(res)   #제목,포스터,리뷰
    # else:
    #     redirect_url = "index.html"

    res = [['기생충1', 'https://image.cine21.com/resize/cine21/poster/2018/1214/11_01_12__5c130ee8e9ae9[X224,320].jpg', '이건리뷰1'],
    ['기생충2', 'https://image.cine21.com/resize/cine21/poster/2018/1214/11_01_12__5c130ee8e9ae9[X224,320].jpg', '이건리뷰2'],
    ['기생충3', 'https://image.cine21.com/resize/cine21/poster/2018/1214/11_01_12__5c130ee8e9ae9[X224,320].jpg', '이건리뷰3'],
    ['기생충4', 'https://image.cine21.com/resize/cine21/poster/2018/1214/11_01_12__5c130ee8e9ae9[X224,320].jpg', '이건리뷰4'],
    ['기생충5', 'https://image.cine21.com/resize/cine21/poster/2018/1214/11_01_12__5c130ee8e9ae9[X224,320].jpg', '이건리뷰5']]

    return render_template(redirect_url, MY_INFO=res)




# @app.route('/result')
# def result():
#     return render_template("result.html", )


if __name__ == '__main__':
    app.debug = True
    app.run(host='127.0.0.1', port=8088)