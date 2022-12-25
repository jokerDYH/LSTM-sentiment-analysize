import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path)

    reviewerID = df['reviewerID']
    flagged = df['flagged']
    restaurantID = df['restaurantID']

    Y_reviewer = {}
    N_reviewer = {}

    Y_restaurant = {}
    N_restaurant = {}
    ##   Y是   N 不是

    for index, i in enumerate(flagged):
        if i == 'Y':  ####  是虚假评论
            if reviewerID[index] not in Y_reviewer.keys():
                Y_reviewer.setdefault(reviewerID[index],{})
                Y_reviewer[reviewerID[index]].setdefault(restaurantID[index],1)
            else:
                if restaurantID[index] in Y_reviewer[reviewerID[index]].keys():
                    Y_reviewer[reviewerID[index]][restaurantID[index]] += 1
                else:
                    Y_reviewer[reviewerID[index]][restaurantID[index]] = 1

            if restaurantID[index] not in Y_restaurant.keys():
                Y_restaurant.setdefault(restaurantID[index],{})
                Y_restaurant[restaurantID[index]].setdefault(reviewerID[index],1)
            else:
                if reviewerID[index] in Y_restaurant[restaurantID[index]].keys():
                    Y_restaurant[restaurantID[index]][reviewerID[index]] += 1
                else:
                    Y_restaurant[restaurantID[index]][reviewerID[index]] = 1

        else:
            if reviewerID[index] not in N_reviewer.keys():
                N_reviewer.setdefault(reviewerID[index],{})
                N_reviewer[reviewerID[index]].setdefault(restaurantID[index],1)
            else:
                if restaurantID[index] in N_reviewer[reviewerID[index]].keys():
                    N_reviewer[reviewerID[index]][restaurantID[index]] += 1
                else:
                    N_reviewer[reviewerID[index]][restaurantID[index]] = 1

            if restaurantID[index] not in N_reviewer.keys():
                N_restaurant.setdefault(restaurantID[index],{})
                N_restaurant[restaurantID[index]].setdefault(reviewerID[index],1)
            else:
                if reviewerID[index] in N_reviewer[restaurantID[index]].keys():
                    N_restaurant[restaurantID[index]][reviewerID[index]] += 1
                else:
                    N_restaurant[restaurantID[index]][reviewerID[index]] = 1

    return Y_reviewer,N_reviewer,Y_restaurant,N_restaurant

def visual(Y_reviewer,xlabel,ylabel):
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    _count = {}
    for key,values in Y_reviewer.items():
        count  = 0
        for value in values.values():
            count += value
        _count[key] = count
    _count = sorted(_count.items(),key=lambda x:x[1])

    xer = [i[0] for i in _count[-16:]]
    yer = [i[1] for i in _count[-16:]]

    fig= plt.figure()
    plt.bar(xer,yer,0.4,color="green")

    plt.xticks(rotation = 90)
    plt.xlabel(f"{xlabel}")
    plt.ylabel(f"{ylabel}")
    plt.title(f"{xlabel}"+f"&{ylabel}——bar chart")
    plt.show()
    plt.savefig('image/q7/'+f"{xlabel}"+f"&{ylabel}——bar chart")

    return fig