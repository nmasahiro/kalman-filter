import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

if __name__ == '__main__':

    df = pd.read_csv("./kf-results.csv")
    T = len(df)

    plt.figure()
    plt.plot(range(T), df["true_x1"], label="true_x1")
    plt.plot(range(T), df["mu1"], label="mu1")
    plt.fill_between(range(T), df["sigma1"], alpha=0.4)
    plt.legend()
    plt.savefig("./kf_x1.pdf")

    plt.figure()
    plt.plot(range(T), df["true_x2"], label="true_x2")
    plt.plot(range(T), df["mu2"], label="mu2")
    plt.fill_between(range(T), df["sigma2"], alpha=0.4)
    plt.legend()
    plt.savefig("./kf_x2.pdf")
