import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# probability_list = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
# ICV_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 1]

if __name__ == '__main__':
    import numpy

    np_timer_speed = numpy.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_speed.csv", "r"),
        delimiter=",", skiprows=0)
    np_timer_step = numpy.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_step.csv", "r"),
        delimiter=",", skiprows=0)

    np_greedy_speed = numpy.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_speed.csv", "r"),
        delimiter=",", skiprows=0)
    np_greedy_step = numpy.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_step.csv", "r"),
        delimiter=",", skiprows=0)

    prob = [0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    icv_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    # X, Y = np.meshgrid(np_prob, np_icv_ratio)
    timer_speed = {}
    timer_step = {}
    greedy_speed = {}
    greedy_step = {}

    for i in range(len(prob)):
        timer_speed[prob[i]] = np_timer_speed[i]
        timer_step[prob[i]] = np_timer_step[i]
        greedy_speed[prob[i]] = np_greedy_speed[i]
        greedy_step[prob[i]] = np_greedy_step[i]
    timer_speed_data = pd.DataFrame(timer_speed, index=icv_ratio, columns=prob)
    timer_step_data = pd.DataFrame(timer_step, index=icv_ratio, columns=prob)
    greedy_speed_data = pd.DataFrame(greedy_speed, index=icv_ratio, columns=prob)
    greedy_step_data = pd.DataFrame(greedy_step, index=icv_ratio, columns=prob)

    plt.figure(1)
    sns.heatmap(timer_speed_data,
                cmap=plt.cm.Blues,
                fmt='.2f',
                vmax=50, vmin=10,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    plt.xlabel("Vehicle Generation Probability")
    plt.ylabel("ICV ratio")
    plt.title("Average Travel Time - Timer")

    plt.savefig("baseline_timer_speed.png", dpi=600)
    plt.show()
    plt.close()

    plt.figure(2)
    sns.heatmap(timer_step_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=5000, vmin=100,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    plt.xlabel("Vehicle Generation Probability")
    plt.ylabel("ICV ratio")
    plt.title("Deadlock Step - Timer")

    plt.savefig("baseline_timer_step.png", dpi=600)
    plt.show()
    plt.close()

    plt.figure(3)
    sns.heatmap(greedy_speed_data,
                cmap=plt.cm.Blues,
                fmt='.2f',
                vmax=50, vmin=10,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    plt.xlabel("Vehicle Generation Probability")
    plt.ylabel("ICV ratio")
    plt.title("Average Travel Time - Greedy")

    plt.savefig("baseline_greedy_speed.png", dpi=600)
    plt.show()
    plt.close()

    plt.figure(4)
    sns.heatmap(greedy_step_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=5000, vmin=100,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    plt.xlabel("Vehicle Generation Probability")
    plt.ylabel("ICV ratio")
    plt.title("Deadlock Step - Greedy")

    plt.savefig("baseline_greedy_step.png", dpi=600)
    plt.show()
    plt.close()
