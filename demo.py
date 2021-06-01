import numpy as np
import math
import matplotlib.pyplot as plt
dt = 0.1  # 采样间隔
NP = 100  # 粒子数
NTh = NP/2
Q = np.diag([0.1])**2  # 协方差
Qsim = np.diag([0.2])**2
MAX_RANGE = 20  # 最大观测范围


def motion_model(x, u):
    B = np.matrix([[dt*math.cos(x[2, 0]), 0.0],
                  [dt*math.sin(x[2, 0]), 0.0],
                  [0.0, dt],
                  [1.0, 0.0]])
    x = x+B.dot(u)
    return x


def pf_filter(xEst, covEst, px, pw, z, u): # px:粒子集合 pw:例子权重
    for i in range(NP):
        pf = motion_model(px[:, i], u)
        w = pw[0, i]
        for j in range(len(z[0, :])):
            zPre = math.sqrt((pf[0, 0]-z[1, j])**2+(pf[1, 0]-z[2, j])**2)
            # sqrt(x^2+y^2)
            dz = zPre-z[0, j]
            # gauss likehood
            sigma = math.sqrt(Q)
            p = 1/math.sqrt(2*math.pi*sigma**2)*math.exp(-dz**2 / (2*sigma**2))
            w = w*p
        px[:, i] = pf[:, 0]
        pw[0, i] = w
    pw = pw/pw.sum()  # gui yi hua
    # estatemate
    xEst = px.dot(pw.T)
    covEst = np.zeros((3, 3))
    for i in range(NP):
        dx = (px[:, i]-xEst)[0:3]
        covEst += pw[0, i]*dx.dot(dx.T)
    # resample
    Neff = 1/pw.dot(pw.T)[0, 0]
    if Neff < NTh:
        wcum = np.cumsum(pw)
        base = np.cumsum(1/NP)-1/NP
        resampleId = base+np.random.rand(base.shape[0])/NP
        inds = []
        ind = 0
        for i in range(NP):
            while resampleId[i] > wcum[ind]:
                ind += 1
            inds.append(ind)
        px = px[:, inds]
        pw = np.zeros((1, NP))+1.0/NP
    return xEst, covEst, px, pw


def get_input(xTrue, xImu, landmark):
    Rsim = np.diag([1.0, math.radians(5.0)])**2
    # calculator volecity
    u = np.matrix([1.0, 0.1]).T
    # jiaosudu he xiansudu
    # calculator new state
    xTrue = motion_model(xTrue, u)  # 无噪声运动模型
    # get measurement
    zt = np.zeros((3, 1))
    for i in range(len(landmark[:, 0])):  # 第i个路标
        dx = landmark[i, 0]-xTrue[0]
        dy = landmark[i, 1]-xTrue[1]
        d = math.sqrt(dx**2+dy**2)
        if d < +MAX_RANGE:
            dn = d+np.random.randn()*0.04  # 随机扰动
            zi = np.matrix([dn, landmark[i, 0], landmark[i, 1]]).T
            if i == 0:
                zt = zi
            else:
                zt = np.hstack((zt, zi))
    ud = np.matrix([u[0, 0]+np.random.randn()*Rsim[0, 0],
                    u[1, 0]+np.random.randn()*Rsim[1, 1]]).T  # 速度扰动
    xImu = motion_model(xImu, ud)
    return ud, zt, xTrue, xImu


if __name__ == "__main__":
    landmark = np.matrix([[10.0, 0.0],
                         [10.0, 10.0],
                         [0.0, 15.0],
                         [-5.0, 20.0]])
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    xImu = xTrue
    covEst = np.eye(4)
    px = np.matrix(np.zeros((4, NP)))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    hTrue = xTrue
    hEst = xEst
    hz = np.zeros((2, 1))
    hImu = xImu
    for i in range(500):
        u, z, xTrue, xImu = get_input(xTrue, xImu, landmark)
        xEst, covEst, px, pw = pf_filter(xEst, covEst, px, pw, z, u)
        hTrue = np.hstack((hTrue, xTrue))
        hEst = np.hstack((hEst, xEst))
        hImu = np.hstack((hImu, xImu))
        # plot
        plt.cla()
        print(len(z[0, :]))
        a = range(len(z[0, :]))
        print(a)
        for i in a:
            plt.plot([xTrue[0, 0], z[1, i]], [xTrue[1, 0], z[2, i]], "-k")
        plt.plot(landmark[:, 0], landmark[:, 1], ".g")
        plt.plot(px[0, :], px[1, :], ".r")
        plt.plot(np.array(hTrue[0, :]).flatten(),
                 np.array(hTrue[1, :]).flatten(), "-b")
        plt.plot(np.array(hEst[0, :]).flatten(),
                 np.array(hEst[1, :]).flatten(), "-r")
        plt.plot(np.array(hImu[0, :]).flatten(),
                 np.array(hImu[1, :]).flatten(), "-k")
        plt.pause(0.001)
