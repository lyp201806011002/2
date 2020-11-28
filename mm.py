
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
import pandas as pd


df = pd.read_excel('data2.xlsx')  # 读取xlsx
data = df.iloc[0:, 1:11].values
states = ["五代十国", "宋", "明 洪武", "明 永乐", "明 宣德", "明 成化", "明 弘治", "明 正德", "明 嘉靖", "明 隆庆", "明 万历", "明 崇祯", "清", "清 康熙",
          "清 雍正", "清 乾隆"]
n_states = len(states)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]

    for n in range(1, len(series)+1):

        if n == 1:

            level, trend = series[0], series[1] - series[0]

        if n >= len(series): # forecasting

            value = result[-1]

        else:

            value = series[n]

    last_level, level = level, alpha * value + (1 - alpha) * (level + trend)

    trend = beta * (level - last_level) + (1 - beta) * trend

    result.append(level + trend)

    return result



def plot_double_exponential_smoothing(series, alphas, betas):

    plt.figure(figsize=(17, 8))

    for alpha in alphas:

        for beta in betas:

            plt.plot(double_exponential_smoothing(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta))

            #plt.plot(series)

            plt.legend(loc="best")

            plt.axis('tight')

            plt.title("Double Exponential Smoothing")

            plt.grid(True)
    plt.show()

plot_double_exponential_smoothing(data,  alphas=[0.9, 0.02], betas=[0.9, 0.02])