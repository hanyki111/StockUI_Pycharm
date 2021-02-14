from bs4 import BeautifulSoup
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kss #문장분리기
from eunjeon import Mecab
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import csv

#네이버 주요 뉴스 크롤링

date = '2021-02-02'
url = 'https://finance.naver.com/news/mainnews.nhn?date=' + date

response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'})
html = response.text
soup = BeautifulSoup(html, 'html.parser')
items = soup.select("ul.newsList li")

article_list = []
article_index_list = [] #article_index_list : 1개 인덱스로 각 뉴스를 뽑아내기 위함. 뉴스가 아닌 것들이 들어가는 경우가 있고 그 경우는 len이 1이므로 각 인덱스 당 len이 1인 경우 삭제 필요
for i in range(len(items)):
    href_start = str(items[i]).find('href') + 6
    href_end_plus = str(items[i])[href_start:].find("\">")

    url_item = str(items[i])[href_start : href_start + href_end_plus]
    url_item = ('https://finance.naver.com' + url_item).replace("&amp;", '&') #&가 &amp;로 되어 url 제대로 인식 불가

    response = requests.get(url_item, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'})
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    article_head = soup.select("div.article_header h3")[0].get_text().replace("\t", "").replace("\n", "").strip() # \t, \n, 문자열 앞뒤 공백 제거
    article_body = soup.select("div.articleCont")[0].get_text()

    article_body_start = article_body.find("기자") + 3
    article_body_end = article_body.find("@")

    article_body = article_body[article_body_start : article_body_end]

    article_body = article_body.replace('…', " ")
    article_body = article_body.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]", "")

    article_list.append(article_head)
    article_temp_list = []
    article_index_list.append([])
    for sent in kss.split_sentences(article_body):
        article_list.append(sent)
        article_temp_list.append(sent)
    article_index_list[i] = article_temp_list

article_index_temp_list = [] #del 관련 문제가 있어 임시 리스트를 생성하여 복제 & 지우는 방식
for i in range(len(article_index_list)):
    if len(article_index_list[i]) == 1 or len(article_index_list[i]) == 0:
        pass
    else:
        article_index_temp_list.append(article_index_list[i])
article_index_list = article_index_temp_list
del article_index_temp_list




#주요 뉴스 전처리 및 모델 업데이트

''' word2vec 모델 업데이트'''
stopwords = ['의', '가', '이', '은', '들', '는', '좀', '잘', '걍', '과', '도', '를', '으로', '자', '에', '와', '로', '을',
             '한', '하다', '이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나', '사람', '는',
             '주', '아니', '등', '같', '우리', '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하',
             '웹사이트', '캡처', '(', ')', '다', '다고', '적', '라고', '"', '[', ']', '/', '=', '·', ',']

mecab = Mecab(dicpath='C:/mecab/mecab-ko-dic')
tokenized_data = []

len_txt = len(article_list)
now_pos = 0

for sentence in article_list:
    now_pos += 1
    progress = round((now_pos / len_txt) * 100)
    if now_pos % 10000 == 1:
        print("텍스트 전처리 " + str(progress) + "% 진행 중")
    temp_X = mecab.morphs(sentence)

    temp_X = [word for word in temp_X if not word in stopwords]
    #print(temp_X)
    if temp_X != []:
        tokenized_data.append(temp_X)

model = Word2Vec.load('news_word2vec')
model.build_vocab(tokenized_data, update=True, min_count=3)

#해당 날짜 주요 뉴스 감정 평가 : 해당 부분 미사용 거의 확정. 정확도 낮음
'''
import json

with open("SentiWord_info.json", 'r', encoding='UTF-8') as senti_json:
    sentiword_list = json.load(senti_json)

sentiword_dict = {}
for i in range(len(sentiword_list)):
    sentiword_dict[sentiword_list[i]['word']] = [sentiword_list[i]['word'], sentiword_list[i]['word_root'], sentiword_list[i]['polarity']]

#단어 in 문장(뉴스 리스트)을 위해 article_index_list 를 문장 단위로 합친 리스트 생성
article_index_scored_list = []
for i in range(len(article_index_list)):
    text = ''.join(article_index_list[i])
    senti_score = 0
    for keys in sentiword_dict.keys():
        if keys in text:
            senti_score += int(sentiword_dict[keys][2])
    article_index_scored_list.append([text, senti_score])

'''

#주요 뉴스 주요 키워드 추출

#해당 키워드 벡터 연산

#연산 벡터와 각 회사 이름들간 벡터 연산

#해당 뉴스의 긍정 / 부정, 가까운 회사 출력

#키워드 사전 추가 필요 : 빅카인즈 -> 공공데이터 개방 -> 데이터목록 : 고빈도사용명사 -> 다운로드 후 사전추가