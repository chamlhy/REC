
# coding: utf-8

# In[1]:


import math
import sys
#from texttable import Texttable


# ## 基于用户的协同过滤
# 
# 基于用户的协同过滤算法主要包括两个步骤：
# * 计算用户之间的相似度
# * 将相似用户喜欢的但目标用户没有听说过的物品推荐给目标用户

# ### 计算用户相似度
# 
# 1. 欧氏距离
# $$\sqrt{\sum_{i=1}^{k}(x_{i}-y_{i})^2}$$
# 2. 余弦相似度（Cosine Similarity）
# $$\frac{X\cdot Y}{\lVert X \rVert \lVert Y \rVert}$$
# 3. Jaccard Similarity
# $$\frac{X \cap Y}{X \cup Y}$$
# 4. 皮尔森相关系数(Pearson Correlation Coefficient)
# 
#  两个连续变量(X,Y)的pearson相关性系数(Px,y)等于它们之间的协方差cov(X,Y)除以它们各自标准差的乘积(σX,σY)。系数的取值总是在-1.0到1.0之间，接近0的变量被成为无相关性，接近1或者-1被称为具有强相关性。
# $$\frac{ n\sum xy - \sum x \sum y}{\sqrt{n\sum x^2 - (\sum x)^2}\sqrt{ n\sum y^2 - (\sum y)^2}}$$

# In[3]:


#求皮尔森相关系数的函数
def pearson1(X, Y):
    n = len(X)
    sum_x = sum([float(X[i]) for i in range(n)])
    sum_y = sum([float(Y[i]) for i in range(n)])
    sum_x_pow = sum([pow(x, 2.0) for x in X])
    sum_y_pow = sum([pow(y, 2.0) for y in Y])
    sum_xy = sum([X[i]*Y[i] for i in range(n)])
    num = n*sum_xy - sum_x*sum_y
    den = math.sqrt(n*sum_x_pow - pow(sum_x, 2.0))*math.sqrt(n*sum_y_pow - pow(sum_y, 2.0))
    if den == 0:
        return 0.0
    return num/den

X = [2,7,18,88,157,90,177,570]

Y = [3,5,15,90,180,88,160,580]

pearson1(X, Y)
                                                    


# ### 计算x和y的相似度
# 用皮尔森相似系数计算 可以写为
# $$sim_{x,y}=\frac {\sum_{i \in I_{xy}}(r_{x,i}- \bar{r}_{x})(r_{y,i}- \bar{r}_{y})}{\sqrt{\sum_{i \in I_{xy}}(r_{x,i}-\bar{r}_{x})^2 \sum_{i \in I_{xy}}(r_{y,i}-\bar{r}_{y})^2}}$$

# In[4]:


#皮尔森相关系数的函数
def pearson2(x, y):
    n = len(x)
    sum_x = sum([float(x[i]) for i in range(n)])
    sum_y = sum([float(y[i]) for i in range(n)])
    avg_x = sum_x/n
    avg_y = sum_y/n
    num = sum([(x[i]-avg_x)*(y[i]-avg_y) for i in range(n)])
    gen = math.sqrt(sum([pow(x[i]-avg_x, 2.0) for i in range(n)])*sum(pow(y[i]-avg_y, 2.0) for i in range(n))*1.0)
    if gen == 0:
        return 0.0
    return num/gen


# ### 用户x对物品i的感兴趣程度
# 
# $$p(u,i) = \sum_{y \in S(x,K) \cap N(i)}sim_{x,y}r_{y,i}$$
# 
# 其中，S(u,K)包含和用户x兴趣最接近的K个用户，N(i)是对物品i有过行为的用户集合，sim是用户x和用户y的兴趣相似度，r是用户y对物品i的兴趣。
# 
# ### 代码实现
# 
# 数据集 ml-100k。若用户A对30部电影打分了，首先求出他打分的平均值，然后高于这个平均值的我们觉得用户喜欢这个电影，否则认为他不喜欢。
# 则有
# $$p(u,i) = \bar{r}_{x} + \frac{\sum_{y \in S(x,K) \cap N(i)}sim_{x,y}(r_{y,i}- \bar{r}_{y})}{\sum_{y \in S(x,K) \cap N(i)}sim_{x,y}}$$

# In[5]:


def load_data(train_file, test_file):
    """读取文件
    param train_file：训练集文件名称
    param test_file：测试集文件名称
    return: 训练集数据，测试集数据和用户用户共有电影矩阵
    """
    train_set = {}
    test_set = {}
    movie_user = {}
    u2u = {}
    
    #加载训练集
    for line in open(train_file):
        (user_id, item_id, rating, timestamp) = line.strip().split('\t')
        train_set.setdefault(user_id, {})
        train_set[user_id].setdefault(item_id, float(rating))
        
        movie_user.setdefault(item_id,[])
        movie_user[item_id].append(user_id.strip())
    
    #加载测试集
    for line in open(test_file):
        (user_id, item_id, rating, timestamp) = line.strip().split('\t')
        test_set.setdefault(user_id, {})
        test_set[user_id].setdefault(item_id, float(rating))
    
    #计算用户与用户共有矩阵
    for m in movie_user.keys():
        for u in movie_user[m]:
            u2u.setdefault(u, {})
            for v in movie_user[m]:
                if u != v:
                    u2u[u].setdefault(v,[])
                    u2u[u][v].append(m)
    
    return train_set,test_set,u2u

def get_avg_rating(user, data_set):
    """获得用户的平均评分
    param user：用户的id
    param data_set: 数据集
    return：用户对所有评价过的电影的平均评分
    """
    average = (sum(data_set[user].values())*1.0) / len(data_set[user].keys())    
    return average

def get_users_sim(u2u, data_set):
    """计算用户之间的相似度
    param u2u：用户与用户共有电影矩阵
    param data_set：数据集
    return：用户与用户相似度矩阵
    """
    users_sim = {}
    for u in u2u.keys():
        users_sim.setdefault(u, {})
        avg_u2movie = get_avg_rating(u, data_set)
        for v in u2u[u]:
            users_sim[u].setdefault(v, 0)
            avg_v2movie = get_avg_rating(v, data_set)
            u_rating = []
            v_rating = []
            for m in u2u[u][v]:
                u_rating.append(data_set[u][m])
                v_rating.append(data_set[v][m])
            users_sim[u][v] = pearson2(u_rating, v_rating)
    return users_sim

def get_rec(N, data_set, users_sim):
    """寻找N个最相似的用户并返回推荐结果
    param N：选择的相似用户数
    param dataset：数据集
    param users_sim：用户相似度矩阵
    return：用户对电影的喜爱度矩阵
    """
    pred = {}
    for u in data_set.keys():
        pred.setdefault(u, {})
        interacted_items = data_set[u].keys() #获取该用户评过分的电影    
        average_u_rate = get_avg_rating(u, data_set)  #获取该用户的评分平均分
        userSimSum = 0
        sim_users = sorted(users_sim[u].items(), key = lambda x: x[1], reverse = True)[0:N]
        
        for v, sim in sim_users:
            average_v_rate = get_avg_rating(v, data_set)
            userSimSum += sim   #对该用户近邻用户相似度求和
            for m, rating in data_set[v].items():
                if m in interacted_items:
                    continue
                else:
                    pred[u].setdefault(m, 0)
                    pred[u][m] += sim*(rating-average_v_rate)
        for m in pred[u].keys():
            pred[u][m] = average_u_rate + (pred[u][m]*1.0) / userSimSum
    
    return pred

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
        


# ### TopN推荐
# 根据评分预测推荐用户最感兴趣的N个电影
# 
# 评价参数：
# * 准确率
# $$precision = \frac{\sum_{u \in U}\lvert R(u)\cap T(u) \rvert}{\sum_{u \in U}\lvert R(u) \rvert}$$
# * 召回率
# $$recall = \frac{\sum_{u \in U}\lvert R(u)\cap T(u) \rvert}{\sum_{u \in U}\lvert T(u) \rvert}$$
# 注：R(u)是根据用户在训练集上的行为给用户做出的推荐列表，T(u)是用户在测试集上的行为列表。

# In[18]:


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



def getMoviesList(file_name):
    #print sys.getdefaultencoding()
    movies_contents=readFile(file_name)
    movies_info={}
    for movie in movies_contents:
        movie_info=movie.split("|")
        movies_info[int(movie_info[0])]=movie_info[1:]
    return movies_info


# In[19]:


#跑一下看看
train_set,test_set,u2u = load_data('./ml-100k/u1.base','./ml-100k/u1.test')
users_sim = get_users_sim(u2u,train_set)

for N in (5,10,20,30,40,50,60,70,80,90,100):            #对不同的近邻数  
    pred = get_rec(N,train_set,users_sim)   #获得推荐  
    mae = get_MAE(test_set,pred)  #计算MAE  
    print('邻居数为：N= %d 时 预测评分误差为：MAE=%f'%(N,mae))
    top_n = get_rec_n(pred, 20)
    precision = get_precision(top_n,test_set)
    recall = get_recall(top_n, test_set)
    print('准确率：%f ,召回率为：%f'%(precision*100.0, recall*100.0))


# ### 总结
# 单纯的user-based协同过滤推荐算法的准确率和召回率并不高
