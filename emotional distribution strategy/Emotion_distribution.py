# coding = utf-8
'''
This code is for emotional score generation and emotional distriution genenration
New version: developed by Chenglin Jiang (cljiangjiang@gmail.com)
'''
import jieba
import re
import json
import tarfile
import configparser
import pickle
import os
import csv
import time
import datetime
import random
import json
import math
import warnings
from collections import Counter
from math import sqrt
from tensorflow.python.platform import gfile
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
# (1)icount 还没改，要限制在同一句表达中
# (2)多个以句号分句的句子可能不会成功。而且还未查找更具体的断句字符。
# ***********************加载EL****************************
int2emotion = ['others', 'like', 'sad', 'disgust', 'angry', 'happy']


nn = ['sad','angry', 'disgust']
pp = ['like', 'happy']
la = []

# a-f 依次对应list里的情感
lines = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/other.txt', encoding='utf-8').read().splitlines()
a = [x for x in lines]
la.append(a)
lines1 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c1.txt', encoding='utf-8').read().splitlines()
b = [x for x in lines1]
la.append(b)
lines2 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c2.txt', encoding='utf-8').read().splitlines()
c1 = [x for x in lines2]
la.append(c1)
lines3 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c3.txt', encoding='utf-8').read().splitlines()
d = [x for x in lines3]
la.append(d)
lines4 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c4.txt', encoding='utf-8').read().splitlines()
e = [x for x in lines4]
la.append(e)
lines5 = open('/Users/apple/Desktop/DLUT-Emotionontology-master/分离文件/分类后/c5.txt', encoding='utf-8').read().splitlines()
f = [x for x in lines5]
la.append(f)

#打开词典文件，返回列表
def open_dict(Dict = 'hahah', path=r'/Users/apple/Desktop/Textming/Sent_Dict/Hownet/'):
    path = path + '%s.txt' % Dict
    dictionary = open(path, 'r', encoding='utf-8')
    dict = []
    for word in dictionary:
        word = word.strip('\n')
        dict.append(word)
    return dict


def judgeodd(num):
    if (num % 2) == 0:
        return 'even'
    else:
        return 'odd'


#注意，这里你要修改path路径。
deny_word = open_dict(Dict = '否定词', path= r'/Users/apple/Desktop/Textming/')
posdict = open_dict(Dict = 'positive', path= r'/Users/apple/Desktop/Textming/')
negdict = open_dict(Dict = 'negative', path= r'/Users/apple/Desktop/Textming/')

degree_word = open_dict(Dict = '程度级别词语', path= r'/Users/apple/Desktop/Textming/')
mostdict = degree_word[degree_word.index('extreme')+1 : degree_word.index('very')]#权重4，即在情感词前乘以4
verydict = degree_word[degree_word.index('very')+1 : degree_word.index('more')]#权重3
moredict = degree_word[degree_word.index('more')+1 : degree_word.index('ish')]#权重2
ishdict = degree_word[degree_word.index('ish')+1 : degree_word.index('last')]#权重0.5



def sentiment_score_list(dataset):
    seg_sentence = [dataset]

    count1 = []
    count2 = []
    for sen in seg_sentence: #循环遍历每一个评论
        segtmp = jieba.lcut(sen, cut_all=False)  #把句子进行分词，以列表的形式返回
        # print(segtmp)
        i = 0 #记录扫描到的词的位置
        a = 0 #记录情感词的位置

        for word in segtmp:
            poscount = 0  # 积极词的第一次分值
            poscount2 = 0  # 积极词反转后的分值
            poscount3 = 0  # 积极词的最后分值（包括叹号的分值）
            negcount = 0
            negcount2 = 0
            negcount3 = 0
            if word in posdict:  # 判断词语是否是情感词
                # print('active: ', word)
                poscount += 1
                c = 0
                a = i + 1
                # print('a', a)
                for w in segtmp[:a][::-1]:  # 扫描情感词前的程度词
                    # print('w',w)
                    if w == "，" or w == ' ，' or w == '， ' :
                        break
                    if w in mostdict:
                        poscount *= 4.0
                    elif w in verydict:
                        poscount *= 3.0
                    elif w in moredict:
                        poscount *= 2.0
                    elif w in ishdict:
                        poscount *= 0.5
                    elif w in deny_word:
                        c += 1

                if judgeodd(c) == 'odd':  # 扫描情感词前的否定词数
                    poscount *= -1.0
                    poscount2 += poscount
                    poscount = 0
                    poscount3 = poscount + poscount2 + poscount3
                    poscount2 = 0
                else:
                    poscount3 = poscount + poscount2 + poscount3
                    poscount = 0
                # a = i + 1  # 情感词的位置变化

            elif word in negdict:  # 消极情感的分析，与上面一致
                # print('negtive: ', word)
                negcount += 1
                d = 0
                a = i + 1
                # print('a',a)
                for w in segtmp[:a][::-1]:  # 扫描情感词前的程度词
                    if w == "，" or w == ' ，' or w == '，':
                        break

                    if w in mostdict:
                        negcount *= 4.0
                    elif w in verydict:
                        negcount *= 3.0
                    elif w in moredict:
                        negcount *= 2.0
                    elif w in ishdict:
                        negcount *= 0.5
                    elif w in degree_word:
                        d += 1

                if judgeodd(d) == 'odd':
                    negcount *= -1.0
                    negcount2 += negcount
                    negcount = 0
                    negcount3 = negcount + negcount2 + negcount3
                    negcount2 = 0
                else:
                    negcount3 = negcount + negcount2 + negcount3
                    negcount = 0

            elif word == '！' or word == '！':  ##判断句子是否有感叹号
                # print('发现感叹号===-===')
                for w2 in segtmp[:segtmp.index(word)][::-1]:  # 扫描感叹号前的情感词，发现后权值+2，然后退出循环
                    # print('w2  ', w2)
                    if w2 =="，" or w2 ==' ，' or w2 =='， ' :
                        # print('break')
                        break


                    if w2 in posdict:

                        poscount3 += 2


                    if w2 in negdict:
                        # print('w2  ', w2)

                        negcount3 += 2

            i += 1 # 扫描词位置前移


            # 以下是防止出现负数的情况
            pos_count = 0
            neg_count = 0
            if poscount3 < 0 and negcount3 >= 0:
                neg_count += negcount3 - poscount3
                pos_count = 0
            elif negcount3 < 0 and poscount3 >= 0:
                pos_count = poscount3 - negcount3
                neg_count = 0
            elif poscount3 < 0 and negcount3 < 0:
                neg_count = -poscount3
                pos_count = -negcount3
            else:
                pos_count = poscount3
                neg_count = negcount3

            count1.append([pos_count, neg_count])
        count2.append(count1)


        count1 = []

    return count2

def sentiment_score(senti_score_list):
    score = []
    # print(senti_score_list)
    for review in senti_score_list:
        # print('reviews:  ', review)
        score_array = np.array(review)
        Pos = np.sum(score_array[:, 0])
        # print('hello')
        Neg = np.sum(score_array[:, 1])
        AvgPos = np.mean(score_array[:, 0])
        AvgPos = float('%.1f'%AvgPos)
        AvgNeg = np.mean(score_array[:, 1])
        AvgNeg = float('%.1f'%AvgNeg)
        StdPos = np.std(score_array[:, 0])
        StdPos = float('%.1f'%StdPos)
        StdNeg = np.std(score_array[:, 1])
        StdNeg = float('%.1f'%StdNeg)
        score.append([Pos, Neg, AvgPos, AvgNeg, StdPos, StdNeg])
    return score

def distribution(sentence, label, r):
    score1 = sentiment_score(sentiment_score_list(sentence))
    #print('score = ', score1)
    p = 0
    q = 0
    score = score1[0]
    if score[0] ==0 and score[1] ==0:
        p = score[0]/(score[0]+score[1])
        #print('p是：', p)
        q = score[4]/(score[4]+score[5])
    e_word = []
    init = [0,0,0,0,0,0]
    init[label] = 1
    count = [0,0,0,0,0,0]
    distr = [0, 0, 0, 0, 0, 0]
    segs = jieba.lcut(sentence, cut_all=False)
    for word in segs:
        if word in a:
            init[0] = 1
            count[0] += 1
            e_word.append(word)
            e_word.append(word)
        elif word in b:
            init[1] = 1
            count[1] += 1
            e_word.append(word)
        elif word in c1:
            init[2] = 1
            count[2] += 1
            e_word.append(word)
        elif word in d:
            init[3] = 1
            count[3] += 1
            e_word.append(word)
        elif word in e:
            init[4] = 1
            count[4] += 1
            e_word.append(word)
        elif word in f:
            init[5] = 1
            count[5] += 1
            e_word.append(word)

    if np.sum(init, 0) == 0:
        print('sum == 0/1')
        init[label] =1
        return init
    if np.sum(init, 0) == 1:
        return init




    if np.sum(init, 0) > 1 and init[label] ==1:
        print('np.sum(init, 0) > 1 and init[label] ==1')
        u = 0
        v = 0
        for w in e_word:
            if w in posdict:
                u+=1
            if w in negdict:
                v+=1
#A
        if u ==0 or v ==0:
            print('u ==0 or v ==0')
#a
            if np.sum(init, 0) ==2:
                ban = init
                ban[label]=0
                init[label] = r +(1-r)*(count[label]/(count[label]+ count[ban.index(1)]))
                init[ban.index(1)] = 1 - init[label]
                return init
#b
            if np.sum(init, 0) > 2:
                ban = init
                ban[label]=0
                all = 0
                for c in range(6):
                    if init[c] ==1:
                        all+=count[c]
                init[label] = r + (1-r)*(count[label]/all)
                for k in range(6):
                    if init[k] ==1:
                        ban[k] = (1-init[label]) * (count[k]/all)

                ban[label] = init[label]
                return ban
#B
        if u!=0 and v!=0:
            print('u!=0 and v!=0')
#a
            if np.sum(init, 0) ==2:

                for i in init:
                    if i != label and  i==1:
                        init[init.index(i)] = 1-p
                init[label] = p
                return init
#b
            if np.sum(init, 0) > 2:
                print('np.sum(init, 0) > 2')

                postip = (1-r)*q
                negtip = 1-r - postip
                # print('postip: ',postip)
                # print('negtip', negtip)
                pall =0
                nall =0
                for t in pp:
                    if t != int2emotion[label]:
                        pall += count[int2emotion.index(t)]

                for l in nn:
                    if l != int2emotion[label]:
                        nall += count[int2emotion.index(l)]
                # print(pall)
                # print(nall)
                for i,j in enumerate(init):
                    # print(i)
                    # print(j)
                    if j==1 and i != label and int2emotion[i] in pp:
                        # print('count', count[i])
                        init[i] = postip * (count[i]/pall)
                    if j==1 and i != label and int2emotion[i] in nn:
                        # print('count', count[i])
                        init[i] = negtip * (count[i]/nall)

                init[label] = r
                return init

#cond.4
        if np.sum(init, 0) > 1 and init[label] == 0:
            print('np.sum(init, 0) > 1 and init[label] == 0')
            u = 0
            v = 0
            for w in e_word:
                if w in posdict:
                    u += 1
                if w in negdict:
                    v += 1
            # A
            if u == 0 or v == 0:
                # a
                if np.sum(init, 0) == 2:
                    ban = init
                    ban[label] = 0
                    init[label] = r
                    init[ban.index(1)] = 1 - init[label]
                    return init
                # b
                if np.sum(init, 0) > 2:
                    ban = init
                    ban[label] = r
                    all = 0
                    for c in range(6):
                        if init[c] == 1 and c!=label:
                            all += count[c]

                    for k in range(6):
                        if ban[k] == 1:
                            ban[k] = (1 - init[label]) * (count[k] / all)


                    return ban
            # B
            if u != 0 and v != 0:
                # a
                if np.sum(init, 0) == 2:
                    init[label] = p
                    for i in init:
                        if i != label and init[i] == 1:
                            init[i] = 1 - p
                    return init
                # b
                if np.sum(init, 0) > 2:
                    # print('np.sum(init, 0) > 2')

                    postip = (1 - r) * q
                    negtip = 1 - r - postip
                    # print('postip: ', postip)
                    # print('negtip', negtip)
                    pall = 0
                    nall = 0
                    for t in pp:
                        if t != int2emotion[label]:
                            pall += count[int2emotion.index(t)]

                    for l in nn:
                        if l != int2emotion[label]:
                            nall += count[int2emotion.index(l)]
                    # print(pall)
                    # print(nall)
                    for i, j in enumerate(init):
                        print(i)
                        print(j)
                        if j == 1 and i != label and int2emotion[i] in pp:
                            print('count', count[i])
                            init[i] = postip * (count[i] / pall)
                        if j == 1 and i != label and int2emotion[i] in nn:
                            print('count', count[i])
                            init[i] = negtip * (count[i] / nall)

                    init[label] = r
                    return init

# rr = 0.6
# data2 = '好像 是 天天 那时 人 小 还 不大 懂事'
# distributionss = distribution(data2, 0, rr)
# print(distributionss)



lidik = open('/Users/apple/Desktop/Textming/cate1s', 'w')
lidib = open('/Users/apple/Desktop/Textming/cate1e', 'w')
lili = []
lidibb = []
ll = []
dataar = []
dev = json.load(open('/Users/apple/Desktop/dev', encoding='utf-8'))
# print(dev[1][1][0].replace('\n',' '))
# a = np.random.choice(dev, 5000)
# print(len(a))
cc = 0
for i in range(5000):
    tim = random.randint(0,10000)
    if tim == cc:
        tim = random.randint(0, 10000)
    else:
        data2 = dev[tim][1][0].replace('\n',' ')
        labela = int(dev[tim][1][1])
        rr = 0.6

        distributionss = distribution(data2, labela, rr)
        k = 0
        if distributionss!=None:
            for i in distributionss:
                if i ==0:
                    k+=1
            if k !=5 and len(lili)<101:
                lili.append(data2)
                lidibb.append(labela)
                nvm = str(distributionss)
                ll.append(nvm)

print('结束1')
dataar.append(lili)
dataar.append(lidibb)
dataar.append(ll)
np_data = np.array(dataar)
np_data = np_data.T
np.array(np_data)
save = pd.DataFrame(np_data, columns=['sentence', 'emotion label', 'distribution'])
save.to_csv('/Users/apple/Desktop/Textming/EDA.csv')

lidik.write(json.dumps(lili, ensure_ascii=False))
lidib.write(json.dumps(lidibb, ensure_ascii=False))

# df = pd.read_csv('/Users/apple/Desktop/textClassifier-master/data/preProcess/ecm_test_data_with_label.csv')
# labels = df["sentiment"].tolist()
# review = df["review"].tolist()
# cc=0
# lk = []
# le = []
# ll = []
# dataar =  []
# for i in range(5000):
#     tim = random.randint(0,len(review))
#     if tim == cc:
#         tim = random.randint(0, len(review))
#     else:
#         data2 = review[i]
#         labela = labels[i]
#         rr = 0.6
#
#         distributionss = distribution(data2, labela, rr)
#         k = 0
#         if distributionss!=None:
#             for i in distributionss:
#                 if i ==0:
#                     k+=1
#             if k !=5 and len(lk)<100:
#                 lk.append(data2)
#                 le.append(labela)
#                 nvm = str(distributionss)
#                 ll.append(nvm)
#                 # v.write(data2+"    "+nvm)setting an array element with a sequence
#                 # v.write('\n')
# dataar.append(lk)
# dataar.append(le)
# dataar.append(ll)
# np_data = np.array(dataar)
# np_data = np_data.T
# np.array(np_data)
# save = pd.DataFrame(np_data, columns=['sentence', 'emotion label', 'distribution'])
# save.to_csv('/Users/apple/Desktop/Textming/EDA.csv')
#
# lidik.write(json.dumps(lk, ensure_ascii=False))
# lidib.write(json.dumps(le, ensure_ascii=False))

#v = open('/Users/apple/Desktop/Textming/EDA1', 'w')
# lis = json.load(open('/Users/apple/Desktop/Textming/cate2s', encoding='utf-8'))
# emo = json.load(open('/Users/apple/Desktop/Textming/cate2e', encoding='utf-8'))
# l1, l2, l3, dataar = [],[],[], []
# rr = 0.6
# for i,j in enumerate(lis):
#     distributionss = distribution(j, emo[i], rr)
#     l1.append(j)
#     l2.append(emo[i])
#     nvm = str(distributionss)
#     l3.append(nvm)
# dataar.append(l1)
# dataar.append(l2)
# dataar.append(l3)
# np_data = np.array(dataar)
# np_data = np_data.T
# np.array(np_data)
# save = pd.DataFrame(np_data, columns=['sentence', 'emotion label', 'distribution'])
# save.to_csv('/Users/apple/Desktop/Textming/EDA1.csv')
# print('over')
# data = '这手机的画面极好，操作也比较流畅，不过拍照真的太烂了！系统也不好'
# data2 = ['我', '太', '开心了','和', '高兴','但是', '我有', '是', '有一点', '伤心','难过','，','我', '不是', '很', '想离开', '，','再见','，','我', '好', '难过', '的','，','会', '快乐的']
# #data2 = '物价上涨压力加大，部分城市房价涨幅过高，涨价太猛，吃不消也'
# # data2 = ''
# data2 = ''.join(data2)
#
# print(data2)
# labela = 2
# rr = 0.5
# distributionss = distribution(data2, labela, rr)
# # print(sentiment_score(sentiment_score_list(data2)))
# print(distributionss)
# dataSource = '/Users/apple/Desktop/会议论文相关材料/新实验/情感词典评分/hotelan/ChnSentiCorp_htl_all.csv'
# df2 = pd.read_csv(dataSource)
# labels = df2["label"].tolist()
# review = df2["review"].tolist()
#
# ee=0
# rr=0
# for i in range(1000):
#
#     print(review[i])
#     score = sentiment_score(sentiment_score_list(review[i]))
#     print(score)
#     score1 = score[0]
#     if score1[0] > score1[1] and labels[i]==1:
#         print('yes')
#         ee+=1
#     if score1[0] < score1[1] and labels[i] == 0:
#         print('yes')
#         rr+=1
# ratio = (ee+rr)/1000
# print('ratio:', ratio)


