import torch

from tool.BertNet import train,word2embedding,get_word,to_csv,get_bert_encoding
from tool.LSTMNet import train_LSTM,predict_sentiment,BiRNN
from tool.Q7 import load_data,visual
from tool.Q8 import input_restaurantID
from tool.Q9 import input_reviewerID
if __name__ == '__main__':
    ###********************************** 问题二 ****************************************#######
    ##  训练bert
    train()

    ###  bert 得到词向量
    bert_model = torch.load(r'model\bert.pth')    ####  加载bert模型
    total_word = get_word(r'data\review_final.csv')  ##  获取全部词汇
    embedding = word2embedding(total_word,bert_model)  ##  词汇转词向量
    to_csv(embedding,total_word)   ###  保存到csv文件

    ###********************************** 问题三 ****************************************#######

    ###  训练LSTM
    train_LSTM(lr=0.001,num_epochs=50)  ###自动保存模型

    ###  预测
    lstm_model  = torch.load('model/LSTM_sentiment_analysis.pth')
    se = "Great food great service !! Even though price is high value is great for the tasty meats"
    pre_result = predict_sentiment(net=lstm_model,sequence=se,rating=4,usefulCount=1,coolCount=0,funnyCount=0)  ###  输出Y/N
    print(pre_result)

    ####********************************** 问题七 ****************************************#######
    ####                      统计虚假评论涉及到的餐厅和评论者，做可视化显示。

    Y_reviewer, N_reviewer, Y_restaurant, N_restaurant = load_data('data/review_final.csv')
    fig_Y_restaurant = visual(Y_restaurant,'虚假餐厅id','虚假评论数量')  ##显示图像
    fig_Y_reviewer = visual(Y_reviewer,'虚假评论id','虚假评论数量')      ##显示图像

    ####********************************** 问题八 ****************************************#######
    input_restaurantID('1iehp7Z2kejBq3RSeb4Dxg')###  输入餐厅id

    ####********************************** 问题九 ****************************************#######
    input_reviewerID('075rcvKMddtsye6OzrFIcg')###  输入评论者id