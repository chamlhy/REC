import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

#之前处理数据集的步骤
#读取文件
names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('./ml-100k/u.data', sep='\t', names=names)
#获取user和item数量
n_users = df.user_id.unique().shape[0]
n_items = df.item_id.unique().shape[0]
#生成评分表
ratings = np.zeros((n_users,n_items))
for row in df.itertuples():  #返回一行元素，row[0]为行的idx
    ratings[row[1]-1,row[2]-1] = row[3]

#分割训练集与测试集 从每个用户的评分中移出10个给测试集
def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    
    for user in range(ratings.shape[0]):
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0], size=10, replace=False) #从不为0的idx中选出10个，且不重复
        
        #分别赋值
        train[user,test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
    
    # 测试是否完全正交
    assert(np.all((train * test) == 0))
    
    return train, test