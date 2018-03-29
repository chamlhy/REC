import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import time

def load_matrix(filename, num_users, num_items):
    t0 = time.time()
    #构建一个稀疏矩阵
    counts = sparse.dok_matrix((num_users, num_items), dtype=float)
    total = 0.0
    num_zeros = num_users * num_items
    #用enumerate同时获得行数和行的内容
    for i, line in enumerate(open(filename, 'r')):
        user, item, count = line.strip().split('\t')
        user = int(user)
        item = int(item)
        count = float(count)
        if user >= num_users:
            continue
        if item >= num_items:
            continue
        if count != 0:
            counts[user, item] = count
            total += count
            num_zeros -= 1
        if i % 100000 == 0:
            print('loaded %i counts...' % i)
    #论文中说alpha=40取得较好结果。这里不同
    alpha = num_zeros / total
    print('alpha %.2f' % alpha)
    counts *= alpha
    #压缩矩阵
    counts = counts.tocsr()
    t1 = time.time()
    print('Finished loading matrix in %f seconds' % (t1 - t0))
    return counts


class ImplicitMF():

    def __init__(self, counts, num_factors=40, num_iterations=30,
                 reg_param=0.8):
        self.counts = counts
        self.num_users = counts.shape[0]
        self.num_items = counts.shape[1]
        self.num_factors = num_factors
        self.num_iterations = num_iterations
        self.reg_param = reg_param

    def train_model(self):
        self.user_vectors = np.random.normal(size=(self.num_users,
                                                   self.num_factors))
        self.item_vectors = np.random.normal(size=(self.num_items,
                                                   self.num_factors))

        for i in range(self.num_iterations):
            t0 = time.time()
            print('Solving for user vectors...')
            #self.item_vectors作为Compressed Sparse Row matrix格式传入，同counts
            self.user_vectors = self.iteration(True, sparse.csr_matrix(self.item_vectors))
            print('Solving for item vectors...')
            self.item_vectors = self.iteration(False, sparse.csr_matrix(self.user_vectors))
            t1 = time.time()
            print('iteration %i finished in %f seconds' % (i + 1, t1 - t0))

    def iteration(self, user, fixed_vecs):
        num_solve = self.num_users if user else self.num_items
        num_fixed = fixed_vecs.shape[0]
        YTY = fixed_vecs.T.dot(fixed_vecs)
        #对角单位矩阵
        eye = sparse.eye(num_fixed)
        lambda_eye = self.reg_param * sparse.eye(self.num_factors)
        solve_vecs = np.zeros((num_solve, self.num_factors))

        t = time.time()
        for i in range(num_solve):
            if user:
                counts_i = self.counts[i].toarray()
            else:
                counts_i = self.counts[:, i].T.toarray()
            #从对角线构造稀疏矩阵
            #因为c_ui = 1 + alpha*r_ui， counts已经*alpha
            #所以这里构建的结果是论文中的C^u-I,所以下文中算YTCupu要+eye
            CuI = sparse.diags(counts_i, [0])
            pu = counts_i.copy()
            #pu为二值矩阵
            pu[np.where(pu != 0)] = 1.0
            YTCuIY = fixed_vecs.T.dot(CuI).dot(fixed_vecs)
            YTCupu = fixed_vecs.T.dot(CuI + eye).dot(sparse.csr_matrix(pu).T)
            #spsolve解决Ax=b，b为向量或矩阵的问题
            xu = spsolve(YTY + YTCuIY + lambda_eye, YTCupu)
            solve_vecs[i] = xu
            if i % 1000 == 0:
                print('Solved %i vecs in %d seconds' % (i, time.time() - t))
                t = time.time()

        return solve_vecs