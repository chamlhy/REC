
# coding: utf-8

# In[1]:


import math


# ### 基于物品的协同过滤
# 
# #### 计算物品相似度
# 
# * 使用余弦相似度计算（同时惩罚热门商品（电影））
# $$sim_{x,y} = \frac{X \cdot Y}{\lVert X \rVert \lVert Y \rVert} = \frac{\sum X_{a,x}Y_{a,y}}{\sqrt{\sum X_{a,x}^2 \sum Y_{a,y}^2}}$$

# In[2]:


def cosine(x, y):
    n = len(x)
    sum_xy = sum([x[i]*y[i] for i in range(n)])
    sum_x2 = sum([pow(x[i], 2.0) for i in range(n)])
    sum_y2 = sum([pow(y[i], 2.0) for i in range(n)])
    gen = math.sqrt(sum_x2*sum_y2)
    if gen == 0:
        return 0
    return sum_xy/gen

X = [2,7,18,88,157,90,177,570]

Y = [3,5,15,90,180,88,160,580]

cosine(X, Y)


# #### 做出预测
# 当为item-based CF做一个推荐时候，你不要纠正用户的平均评价，因为用户本身用查询来做预测。
# 
# 用户u对物品i的感兴趣程度为：
# $$p(u,i) = \frac{\sum_{j \in S(i,K) \cap N(i)}sim_{i,j}r_{u,j}}{\sum_{j \in S(i,K) \cap N(i)}sim_{i,j}}$$
# 
# #### 代码实现
# 数据集 ml-100k。

# In[3]:


def load_data(train_file, test_file):
    """读取文件
    param train_file：训练集文件名称
    param test_file：测试集文件名称
    return: 训练集数据，测试集数据和电影电影共有用户矩阵
    """
    train_set = {}
    test_set = {}
    user_movie = {}
    m2m = {}
    
    #加载训练集
    for line in open(train_file):
        (user_id, item_id, rating, timestamp) = line.strip().split('\t')
        train_set.setdefault(user_id, {})
        train_set[user_id].setdefault(item_id, float(rating))
        
        user_movie.setdefault(user_id,[])
        user_movie[user_id].append(item_id.strip())
    
    #加载测试集
    for line in open(test_file):
        (user_id, item_id, rating, timestamp) = line.strip().split('\t')
        test_set.setdefault(user_id, {})
        test_set[user_id].setdefault(item_id, float(rating))
    
    #计算物品与物品共有矩阵
    for u in user_movie.keys():
        for m in user_movie[u]:
            m2m.setdefault(m, {})
            for n in user_movie[u]:
                if m != n:
                    m2m[m].setdefault(n,[])
                    m2m[m][n].append(u)
    
    return train_set,test_set,m2m

def get_movies_sim(m2m, data_set):
    """计算电影之间的相似度
    param m2m：电影与电影共有用户矩阵
    param data_set：数据集
    return：电影与电影相似度矩阵
    """
    movies_sim = {}
    for m in m2m.keys():
        movies_sim.setdefault(m, {})
        for n in m2m[m]:
            movies_sim[m].setdefault(n, 0)
            m_rating = []
            n_rating = []
            for u in m2m[m][n]:
                m_rating.append(data_set[u][m])
                n_rating.append(data_set[u][n])
            movies_sim[m][n] = cosine(m_rating, n_rating)
    return movies_sim

def get_rec(N, data_set, movies_sim):
    """寻找N个最相似的电影并返回推荐结果
    param N：选择的相似电影数
    param dataset：数据集
    param movies_sim：电影相似度矩阵
    return：用户对电影的喜爱度矩阵
    """
    pred = {}
    for u in data_set.keys():
        pred.setdefault(u, {})
        interacted_items = data_set[u].keys() #用户有过动作的物品集合
        movie_sim_sum = {}
        for m in interacted_items: #对于用户喜欢的每一个电影
            #找到n个与它最相似的电影
            m_sim = sorted(movies_sim[m].items(), key = lambda x :x[1], reverse = True)[0: N]
            tmp_prefect = 0
            for n, sim in m_sim:
                if n in interacted_items:
                    continue
                pred[u].setdefault(n, 0)
                movie_sim_sum.setdefault(n, 0)
                pred[u][n] += sim*data_set[u][m]
                movie_sim_sum[n] += abs(sim)
        for n in pred[u].keys():
            pred[u][n] = pred[u][n]/movie_sim_sum[n]
    return pred


# In[4]:


#根据测试集一个一个计算
def pred1(u, m, movies_sim, data_set, N):
    sim_sum = 0
    pred = 0
    if m not in movies_sim.keys():
        return 0
    m_sim = sorted(movies_sim[m].items(), key = lambda x :x[1], reverse = True)[0: N]
    for n, sim in m_sim:
        if n in data_set[u].keys(): #处于集合之内
            pred += sim*data_set[u][n]
            sim_sum += abs(sim)
    if sim_sum == 0:
        return 0
    return pred/sim_sum

def get_pred(data_set, train_set, movies_sim, N):
    pred = {}
    for u in data_set.keys():
        pred.setdefault(u, {})
        for m in data_set[u].keys():
            pred[u][m] = pred1(u, m,movies_sim, train_set,N)
    return pred


# In[5]:


#评估
def get_MAE(data_set, pred):
    """计算预测评分的准确度
    param data_set：数据集
    param pred：预测矩阵
    return：预测的准确度
    """
    MAE = 0  
    rSum = 0  
    setSum = 0  
  
    for u in pred.keys():    #对每一个用户  
        for m, rating in pred[u].items():    #对该用户预测的每一个电影      
            if u in data_set.keys() and m in data_set[u].keys() :   #如果用户为该电影评过分  
                setSum = setSum + 1     #预测准确数量+1  
                rSum = rSum + abs(data_set[u][m]-rating)      #累计预测评分误差  
    MAE = rSum / setSum  
    return MAE


# In[5]:


#准确率和召回率
def get_rec_n(pred, N):
    """获取前N个推荐
    param pred：评分预测矩阵
    param N：推荐TopN的值
    return 对用户u推荐的top_n列表"""
    top_n = {}
    for u in pred.keys():
        top_n.setdefault(u, [])
        sorted_pred = sorted(pred[u].items(), key = lambda x: x[1], reverse = True)[0:N]
        for m, rating in sorted_pred:
            top_n[u].append(m)
    return top_n

def get_precision(top_n, data_set):
    """获取TopN推荐的准确率
    param top_n：系统给出的TopN推荐
    param data_set：测试数据集
    return：准确率"""
    precision_sum = 0
    u_sum = 0
    for u in top_n.keys():
        if u in data_set.keys():
            u_sum += 1
        pred_sum = 0
        set_sum = len(top_n[u])
        for m in top_n[u]:
            if u in data_set.keys() and m in data_set[u].keys():
                pred_sum += 1
        precision_sum += pred_sum/set_sum
    return precision_sum*1.0/u_sum

def get_recall(top_n, data_set):
    """获取TopN推荐的召回率
    param top_n：系统给出的TopN推荐
    param data_set：测试数据集
    return：召回率"""
    u_sum = 0
    recall_sum = 0
    for u in top_n.keys():
        pred_sum = 0
        if u in data_set.keys():
            u_action = len(data_set[u])
            u_sum += 1
        else:
            u_action = 0
        for m in top_n[u]:
            if u in data_set.keys() and m in data_set[u].keys():
                pred_sum += 1
        if u_action != 0:
            recall_sum += pred_sum/u_action
    return recall_sum*1.0/u_sum


# In[7]:


#跑一下看看
train_set,test_set,m2m = load_data('./ml-100k/u2.base','./ml-100k/u2.test')
movies_sim = get_movies_sim(m2m,train_set)

for N in (5,10,20,30,40):            #对不同的近邻数  
    pred = get_rec(N,train_set,movies_sim)   #获得推荐  
    #mae = get_MAE(test_set,pred)  #计算MAE  
    #print('邻居数为：N= %d 时 预测评分误差为：MAE=%f'%(N,mae))
    top_n = get_rec_n(pred, 20)
    precision = get_precision(top_n,test_set)
    recall = get_recall(top_n, test_set)
    print('准确率：%f ,召回率为：%f'%(precision*100.0, recall*100.0))


# In[6]:


#跑一下看看
train_set,test_set,m2m = load_data('./ml-100k/u2.base','./ml-100k/u2.test')
movies_sim = get_movies_sim(m2m,train_set)

for N in (5,10,20,30,40):
    pred = get_pred(test_set, train_set, movies_sim,N)
    mae = get_MAE(test_set,pred)  #计算MAE  
    print('邻居数为：N= %d 时 预测评分误差为：MAE=%f'%(N,mae))
    #top_n = get_rec_n(pred, 20)
    #precision = get_precision(top_n,test_set)
    #recall = get_recall(top_n, test_set)
    #print('准确率：%f ,召回率为：%f'%(precision*100.0, recall*100.0))


# * 在此数据集上，item-based 协同过滤算法效果也很差
