import torch
import matplotlib.pyplot as plt
import jieba.analyse
import pandas as pd
from wordcloud import WordCloud
from du.tool.LSTMNet import BiRNN,predict_sentiment


def input_restaurantID(restaurantID):
    df = pd.read_csv(r'data/review_final.csv')
    l = list(df['reviewContent'])
    rating = list(df['rating'])
    usefulCount = list(df['usefulCount'])
    coolCount = list(df['coolCount'])
    funnyCount = list(df['funnyCount'])
    flagged = list(df['flagged'])
    restaurantID_total = list(df['restaurantID'])
    index_list = []
    for index,i in enumerate(restaurantID_total):
        if restaurantID == i:
            index_list.append(index)
    if index_list == []:
        print(f"{restaurantID} 餐厅不存在 ")
    else:
        this_restaurant_l = [l[i] for i in index_list]
        this_restaurant_rating = [rating[i] for i in index_list]
        this_restaurant_usefulCount = [usefulCount[i] for i in index_list]
        this_restaurant_coolCount = [coolCount[i] for i in index_list]
        this_restaurant_funnyCount = [funnyCount[i] for i in index_list]
        this_restaurant_flagged = [flagged[i] for i in index_list]


        pre_true_index = []
        model = torch.load('model/LSTM_sentiment_analysis.pth')

        for i in range(len(index_list)):
            # print(f"开始预测第 {index_list[i]}个")
            pre = predict_sentiment(model,this_restaurant_l[i],this_restaurant_rating[i],this_restaurant_usefulCount[i],this_restaurant_coolCount[i],this_restaurant_funnyCount[i])
            if pre== this_restaurant_flagged[i]:
                pre_true_index.append(i)
                # print("预测为相同")
        if pre_true_index == []:
            print(f"{restaurantID}没有真实评论")
        else:
            true_l = []  ###  预测为真的评论
            true_rating = {}   ###   预测为真的评论对应的rating数量   rating：count
            true_l_rating = {}   ###   预测为真的对应的评论    rating：【对应的预测为真的评论 】
            for i in pre_true_index:
                true_l.append(this_restaurant_l[i])  ###   预测为真的评论
                # true_rating.append(this_restaurant_rating[i])    ###  预测为真的 评分
                if this_restaurant_rating[i] not in true_rating.keys():
                    true_rating[this_restaurant_rating[i]] = 1
                else:
                    true_rating[this_restaurant_rating[i]] += 1
                if this_restaurant_rating[i] not in true_l_rating.keys():
                    true_l_rating.setdefault(this_restaurant_rating[i],[]).append(this_restaurant_l[i])
                else:
                    true_l_rating[this_restaurant_rating[i]].append(this_restaurant_l[i])

            plt_mes(true_rating,restaurantID)
            keywords_cloud(true_l,restaurantID)
            mkeywords_cloud(true_l_rating, restaurantID)


####   画图   8-- 2
def plt_mes(true_rating,restaurantID):
    xer = [i for i in true_rating.keys()]
    yer = [true_rating[i] for i in xer]
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    ###  柱状图
    plt.figure()
    plt.bar(xer,yer,0.4,color="green")
    plt.xlabel(f"rating")
    plt.ylabel(f"count")
    plt.title(f"{restaurantID}_rating"+f"&count柱状图")
    plt.savefig("image/q8_2/"+f"{restaurantID}_rating"+f"&count柱状图")
    print("已生成图片"+"image/q8_2/"+f"{restaurantID}_rating"+f"&count柱状图")
    ####  饼图
    election_data = true_rating
    candidate = ['rating' + str(key) for key in election_data]
    votes = [value for value in election_data.values()]
    plt.figure(figsize=(10, 10), dpi=100)
    explode = [0.1 for i in range(len(xer))]
    plt.pie(votes, labels=candidate, autopct="%1.2f%%", colors=['c', 'm', 'y','r','b'], textprops={'fontsize': 24},
        labeldistance=1.05, explode=explode, startangle=90, shadow=True)
    plt.legend(loc='upper right', fontsize=12)
    plt.title("各类rating占比", fontsize=24)
    plt.axis('equal')
    # plt.show()
    plt.savefig("image/q8_2/"+f"{restaurantID}_rating"+f"&count饼图")
    print("已生成图片" + "image/q8_2/"+f"{restaurantID}_rating"+f"&count饼图")

####  8-1
def keywords_cloud(true_l,restaurantID):
    str = " ".join(i for i in true_l)
    keyword = jieba.analyse.extract_tags(str,20)
    wordcloud = WordCloud().generate(' '.join(i for i in keyword))
    wordcloud.to_file(f'image/q8_1/{restaurantID}_keywords_cloud.jpg')
    print("已生成图片"+f'image/q8_1/{restaurantID}_keywords_cloud.jpg')
#### 8-3

def mkeywords_cloud(true_l_rating,restaurantID):

    for key, value in true_l_rating.items():
        strs = " ".join(i for i in value)
        wordcloud = WordCloud().generate(' '.join(i for i in jieba.analyse.extract_tags(strs, 20)))
        wordcloud.to_file(f'image/q8_3/{restaurantID}_rating_{str(key)}_mkeywords_cloud.jpg')
        print("已生成图片" + f'image/q8_3/{restaurantID}_rating_{str(key)}_mkeywords_cloud.jpg')




