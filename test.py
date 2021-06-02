from PaRIS import *


theta = np.zeros(theta_dim) + theta_true + 0.1#np.random.randn(theta_dim)
particle = np.zeros((x_dim, NP))
x = np.zeros((x_dim, T))
y = np.zeros((y_dim, T))
tao = np.zeros((theta_dim, NP)) 
w = np.zeros(NP)

x0 = 100
for i in range(T):
    if i == 0:
        x[:, i] = x0
    else:
        x[:, i] = SM_truth(x[:, i-1], theta_true)  # 真值
    y[:, i] = MM(x[:, i], theta_true)  # 带噪声观测

# set arbitrarily theta0

print(theta)
# draw initial particles
for i in range(NP):
    particle[:, i] = x0 + SM_noise(theta)  # 初始粒子
w = np.zeros(NP) + 1.0 / NP

zeta1 = np.zeros((theta_dim, NP))
zeta2 = np.zeros((theta_dim, NP))
zeta3 = np.zeros(NP)
for t in range(T-1):
    pt, wt, taot = PaRIS(particle, w, tao, y[:, t], y[:, t+1], N_tilde, theta)
    print(t, "is over")guanjian
    tao_bar = np.mean(taot, axis=1)
    for l in range(NP):
        res = derivative_g(pt[:, l], y[:, t+1], theta) * g(pt[:, l], y[:, t+1], theta)
        zeta1[:, l] = res
        res = (taot[:, l] - tao_bar) * g(pt[:, l], y[:, t+1], theta)
        zeta2[:, l] = res
        res = g(pt[:, l], y[:, t+1], theta)
        zeta3[l] = res
    zeta1_hat = np.mean(zeta1, axis=1)
    zeta2_hat = np.mean(zeta2, axis=1)
    zeta3_hat = np.mean(zeta3)
    theta =  theta + 0.0001 * 1.0 / (t+1) * (zeta1_hat + zeta2_hat) / zeta3_hat
    particle, w, tao = pt, wt, taot

print(theta)

#如何设置gamma,即学习率是关键!
