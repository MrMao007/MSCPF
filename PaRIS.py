import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
from scipy.misc import derivative


NP = 100
T = 1000
N_tilde = 2
time_total = 0
trail_count = 0
x_dim = 1
y_dim = 1
theta_dim = 2
tao_dim = theta_dim
theta_true = np.array([0.8, 0.1])


def SM_truth(x, theta):
    x = theta[0] * x
    return x


def SM_noise(theta):
    return np.random.randn()


def SM(x, theta):
    res = SM_truth(x, theta) + SM_noise(theta)
    return res


def q(x, xt, theta):
    res = 1/math.sqrt(2*math.pi*1**2)*math.exp(-(xt - SM_truth(x, theta))**2/(2*1**2))
    return res


def derivative_q(x, xt, theta):  # 对数似然函数求导
    res = np.zeros(theta_dim)
    res[0] = x * (xt- theta[0]*x)
    res[1] = 0
    return res


def MM_truth(x, theta):
    y = theta[1] * x
    return y


def MM_noise(theta):
    return np.random.randn()


def MM(x, theta):
    res = MM_truth(x, theta) + MM_noise(theta)
    return res


def g(x, y, theta):
    # res = 1/math.sqrt(2*math.pi*theta[2])*math.exp(-(y - MM_truth(x, theta))**2/(2*theta[2]))
    res = 1/math.sqrt(2*math.pi*1**2)*math.exp(-(y - MM_truth(x, theta))**2/(2*1**2))
    return res


def derivative_g(x, y, theta):    
    res = np.zeros(theta_dim)
    res[0] = 0
    res[1] = x * (y- theta[1]*x)
    return res


def h_function(x, xt, y, theta):
    dq_theta = derivative_q(x, xt, theta)
    dg_theta = derivative_g(x, y, theta)
    return dq_theta + dg_theta


def resample(weights):  # weight已经归一化
    indices = []
    # 求出离散累积密度函数(CDF)
    # C = [0.] + [sum(weights[:i+1]) for i in range(NP)]
    C = np.cumsum(weights)
    C = np.insert(C, 0, 0.0)
    # base = np.cumsum(1/100)-1/100
    # print(C == C2)
    # 选定一个随机初始点
    u0, j = np.random.random(), 0
    # tmp = [(u0+i)/NP for i in range(NP)];
    for u in [(u0+i)/NP for i in range(NP)]:  # u 线性增长到 1
        while u > C[j]:  # 碰到小粒子，跳过
            j += 1
        indices.append(j-1)  # 碰到大粒子，添加，u 增大，还有第二次被添加的可能
    return indices  # 返回大粒子的下标


def multinomial_sampling(w, n):
    indices = np.zeros(n, dtype='int')
    w = w / np.sum(w)
    q = np.cumsum(w)
    U = np.random.uniform(0, 1, n)
    U = np.sort(U, kind='heapsort')
    sigma = np.random.permutation(n)
    l = 0
    r = 1
    for k in range(n):
        d = 1
        while U[k] >= q[r-1]:  # 寻找U[k]的粗区间,区间长度2**d
            l = r
            r = min(r + 2**d, w.shape[0])
            d = d + 1
        while r - l > 1:      # min r, s.t. U[k] < q[r]
            m = math.floor((l + r)/2.0)
            if U[k] >= q[m-1]:
                l = m
            else:
                r = m
        indices[sigma[k]] = r-1
    return indices

    
def bootstrap_pf(particle, w, y, theta):
    wt = np.zeros(NP)
    pt = np.zeros((x_dim, NP))
    wt = w / np.sum(w)
    indices = resample(wt)
    for i in range(NP):
        pt[:, i] = SM(particle[:, indices[i]], theta)
        # wt[i] = 1/math.sqrt(2*math.pi*1**2)*math.exp(-(y - particle[:, i])**2 / (2*1**2))
        wt[i] = g(pt[:, i], y, theta)
    return pt, wt


def FFBSm(particle, w, tao, y, yt, theta):
    pt, wt = bootstrap_pf(particle, w, yt, theta)
    taot = np.zeros((tao_dim, NP))
    for i in range(NP):
        # print('processing particle', i)
        w_sum = 0
        for j in range(NP):
            # w_sum += w[j]*(1/math.sqrt(2*math.pi*0.2**2)*math.exp(-(particle[:, j]*0.7 - pt[:, i])**2 / (2*0.2**2)))
            w_sum += w[j] * q(particle[:, j], pt[:, i], theta)
        for j in range(NP):
            # h = h_function(particle[:, j], pt[:, i])
            # particlej = particle[:, j]
            # pti = pt[:, i]
            # q = 1/math.sqrt(2*math.pi*0.2**2)*math.exp(-(particlej*0.7 - pti)**2 / (2*0.2**2))
            taot[:, i] += w[j] * q(particle[:, j], pt[:, i], theta) / w_sum * (tao[:, j] + h_function(particle[:, j], pt[:, i], y, theta))
    return pt, wt, taot


def accept_reject_backwark_sampling(particle, w, pt, wt, N_tilde, theta):
    # global time_total
    # global trail_count
    # trail = 0
    epsilon = q(particle[:, 0], SM_truth(particle[:, 0], theta), theta)   # 可能存在问题
    J = np.zeros((NP, N_tilde), dtype='int') + NP
    # return J
    # w_norm = w / np.sum(w)
    for j in range(N_tilde):
        flag = 0
        LAMBDA = np.zeros(NP)
        threshold = math.floor(math.sqrt(NP))
        L = range(NP)  # 0~NP-1
        while L:
            # aj_time_start = time.time()
            n = len(L)
            indices = multinomial_sampling(w, n)
            # indices = bisect.bisect([1,2,3], np.random.random())
            # random.shuffle(indices)
            u = np.random.uniform(0, 1, n)
            Ln = []
            # aj_time_end = time.time()
            # time_total += aj_time_end-aj_time_start
            for k in range(n):
                # q = 1/math.sqrt(2*math.pi*0.2**2)*math.exp(-(particle[0, indices[k]]*0.7 - pt[0, L[k]])**2 / (2*0.2**2))
                if u[k] <= q(particle[:, indices[k]], pt[:, L[k]], theta) / epsilon:
                    J[L[k], j] = indices[k]
                else:
                    Ln.append(L[k])
            L = Ln
            threshold -= 1
            if threshold == 0:
                for i in L:
                    for m in range(NP):
                        LAMBDA[m] = w[i] * q(particle[:, m], pt[:, i], theta)
                    J[i, j] = multinomial_sampling(LAMBDA, 1)
                flag = 1
            # print(np.sum(J == -1))
            # trail += 1
            if flag == 1:
                break
    # print('trail', trail)
    # trail_count += trail
    return J


def PaRIS(particle, w, tao, y, yt, N_tilde, theta):
    # time_total = 0
    # global trail_count
    pt, wt = bootstrap_pf(particle, w, yt, theta)
    taot = np.zeros((tao_dim, NP))
    # trail_count = 0
    J = accept_reject_backwark_sampling(particle, w, pt, wt, N_tilde, theta)
    for i in range(NP):
        # print('processing particle', i)
        # aj_time_start = time.time()
        # aj_time_end = time.time()
        # time_total += aj_time_end-aj_time_start
        # print("accept-reject time:", time_total)
        for j in range(N_tilde):
            taot[:, i] += tao[:, J[i, j]] + h_function(particle[:, J[i, j]], pt[:, i], y, theta)
        taot[:, i] = taot[:, i]/N_tilde
    # print('rj_time', time_total)
    # trail_count = trail_count / float(NP)
    # print('trail count', trail_count)
    return pt, wt, taot


if __name__ == "__main__":
    
    x = np.zeros((x_dim, T))
    y = np.zeros((y_dim, T))
    particle = np.zeros((x_dim, NP))
    x0 = 100  # 粒子初值
    w = np.zeros(NP) + 1.0 / NP
    tao = np.zeros((tao_dim, NP))
    p_his = np.zeros((T, NP))
    w_his = np.zeros((T, NP))
    # tm = np.zeros((1, NP))
    for i in range(NP):
        particle[:, i] = x0 + SM_noise(theta_true)  # 初始粒子
        # tao[0, i] = particle[0, i]
    # indices = resample(w)
    # plt.hist(particle, density=True, label='sampling')
    # plt.show()
    # print(np.var(particle))
    for i in range(T):
        if i == 0:
            x[:, i] = x0
        else:
            x[:, i] = SM_truth(x[:, i-1], theta_true)  # 真值
        y[:, i] = MM(x[:, i], theta_true)  # 带噪声观测
    time_start = time.time()
    for i in range(T-1):
        p_his[i, :] = particle
        w_his[i, :] = w
        # particle, w = bootstrap_pf(particle, w, y[:, i+1])
        pt, wt, taot = PaRIS(particle, w, tao, y[:, i], y[:, i+1], N_tilde, theta_true)
        # pt, wt, taot = FFBSm(particle, w, tao, y[:, i], y[:, i+1], theta_true)
        particle, w, tao = pt, wt, taot
        print(i, "is over")
    p_his[T-1, :] = particle
    w_his[T-1, :] = w
    time_end = time.time()
    
    sum = 0
    for k in range(T-1):
        sum += h_function(x[:, k], x[:, k+1], y[0, k], theta_true)
    est = 0
    est_h = 0
    for i in range(NP):
        est = est + particle[:, i]*w[i]
        est_h += w[i]*tao[:, i]
    est = est / np.sum(w)
    est_h_final = est_h / np.sum(w)
    k = np.sum(w)
    print('truth', sum)
    print('estimated', est_h_final)
    print('totally cost', time_end - time_start)
    
    '''
    w_test = np.zeros(6) + 1/6.0
    indices_test = multinomial_sampling(w_test, 12)
    print(indices_test)
    '''
    # derivative(q, theta, dx=1e-6)
    print("ok")
