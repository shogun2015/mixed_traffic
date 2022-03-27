import numpy as np
import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':

    zhfont = mpl.font_manager.FontProperties(fname='/home/wuth-3090/.local/share/fonts/SimHei.ttf')

    np_timer_time = np.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_uniform_time.csv", "r"),
        delimiter=",", skiprows=0)
    np_timer_step = np.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_uniform_step.csv", "r"),
        delimiter=",", skiprows=0)
    np_timer_wait_lane = np.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/timer_uniform_wait_lane.csv", "r"),
        delimiter=",", skiprows=0)

    np_timer_time = np_timer_time - 15
    np_timer_step = (np_timer_step + 1) / 4

    np_greedy_time = np.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_uniform_time.csv", "r"),
        delimiter=",", skiprows=0)
    np_greedy_step = np.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_uniform_step.csv", "r"),
        delimiter=",", skiprows=0)
    np_greedy_wait_lane = np.loadtxt(
        open("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/greedy_uniform_wait_lane.csv", "r"),
        delimiter=",", skiprows=0)

    np_greedy_time = np_greedy_time - 12
    np_greedy_step = (np_greedy_step + 1) / 4

    np_light_time = np.array([26.12, 35.31, 33.98, 34.45, 43.84, 40.95, 45.31, 57.07, 51.22, 52.98])
    np_light_wait_lane = np.array([11, 13, 13, 25, 31, 31, 31, 31, 31, 31])
    np_light_time = np_light_time - 2

    # prob = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    prob = [36, 72, 108, 144, 180, 216, 252, 288, 324, 360]
    icv_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    # X, Y = np.meshgrid(np_prob, np_icv_ratio)
    timer_timer = {}
    timer_step = {}
    timer_wait_lane = {}
    greedy_timer = {}
    greedy_step = {}
    greedy_wait_lane = {}
    light_time = {}
    light_wait_lane={}

    for i in range(len(prob)):
        timer_timer[prob[i]] = np_timer_time[i]
        timer_step[prob[i]] = np_timer_step[i]
        timer_wait_lane[prob[i]] = np_timer_wait_lane[i]
        greedy_timer[prob[i]] = np_greedy_time[i]
        greedy_step[prob[i]] = np_greedy_step[i]
        greedy_wait_lane[prob[i]] = np_greedy_wait_lane[i]
        light_time[prob[i]] = np_light_time[i]
        light_wait_lane[prob[i]] = np_light_wait_lane[i]

    timer_time_data = pd.DataFrame(timer_timer, index=icv_ratio, columns=prob)
    timer_step_data = pd.DataFrame(timer_step, index=icv_ratio, columns=prob)
    timer_wait_lane_data = pd.DataFrame(timer_wait_lane, index=icv_ratio, columns=prob)
    greedy_time_data = pd.DataFrame(greedy_timer, index=icv_ratio, columns=prob)
    greedy_step_data = pd.DataFrame(greedy_step, index=icv_ratio, columns=prob)
    greedy_wait_lane_data = pd.DataFrame(greedy_wait_lane, index=icv_ratio, columns=prob)
    light_time_data = pd.DataFrame(light_time, index=[1], columns=prob)
    light_wait_lane_data = pd.DataFrame(light_wait_lane, index=[1], columns=prob)

    # ----------------------------------------------------------------------------------

    plt.figure(1)
    sns.heatmap(timer_time_data,
                cmap=plt.cm.Blues,
                fmt='.1f',
                vmax=50, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    # plt.xlabel("Vehicle Generation Probability")
    # plt.ylabel("ICV ratio")
    # plt.title("Average Travel Time - Timer")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.xlabel("车辆产生率（辆/小时）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("平均延迟（s） - 固定时间法", fontproperties=zhfont)

    plt.savefig("baseline_timer_uniform_time.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
    # ==============================================
    plt.figure(2)
    sns.heatmap(timer_step_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=1000, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    # plt.xlabel("Vehicle Generation Probability")
    # plt.ylabel("ICV ratio")
    # plt.title("Deadlock Step - Timer")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.xlabel("车辆产生率（辆/小时）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("“死锁”发生时间（秒） - 固定时间法", fontproperties=zhfont)

    plt.savefig("baseline_timer_uniform_step.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
    # ==============================================
    plt.figure(3)
    sns.heatmap(timer_wait_lane_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=25, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    # plt.xlabel("Vehicle Generation Probability")
    # plt.ylabel("ICV ratio")
    # plt.title("Max Waiting Vehicle on A Lane - Timer")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.xlabel("车辆产生率（辆/小时）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("单车道最大等待车辆数目（辆） - 固定时间法", fontproperties=zhfont)

    plt.savefig("baseline_timer_uniform_wait_lane.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()

    # ----------------------------------------------------------------------------------

    plt.figure(4)
    sns.heatmap(greedy_time_data,
                cmap=plt.cm.Blues,
                fmt='.1f',
                vmax=50, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    # plt.xlabel("Vehicle Generation Probability")
    # plt.ylabel("ICV ratio")
    # plt.title("Average Travel Time - Greedy")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.xlabel("车辆产生率（辆/小时）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("平均延迟（s） - 贪婪法", fontproperties=zhfont)

    plt.savefig("baseline_greedy_uniform_time.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
    # ==============================================
    plt.figure(5)
    sns.heatmap(greedy_step_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=1000, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    # plt.xlabel("Vehicle Generation Probability")
    # plt.ylabel("ICV ratio")
    # plt.title("Deadlock Step - Greedy")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.xlabel("车辆产生率（辆/小时）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("“死锁”发生时间（秒） - 贪婪法", fontproperties=zhfont)

    plt.savefig("baseline_greedy_uniform_step.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
    # ==============================================
    plt.figure(6)
    sns.heatmap(greedy_wait_lane_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=25, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    # plt.xlabel("Vehicle Generation Probability")
    # plt.ylabel("ICV ratio")
    # plt.title("Max Waiting Vehicle on A Lane - Greedy")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.xlabel("车辆产生率（辆/小时）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("单车道最大等待车辆数目（辆）- 贪婪法", fontproperties=zhfont)

    plt.savefig("baseline_greedy_uniform_wait_lane.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()

    # ----------------------------------------------------------------------------------

    # plt.figure(7)
    # sns.heatmap(light_time_data,
    #             cmap=plt.cm.Blues,
    #             fmt='.1f',
    #             vmax=30, vmin=0,
    #             linewidths=0.1,
    #             xticklabels=True, yticklabels=True,
    #             annot=True)
    # # plt.xlabel("Vehicle Generation Probability")
    # # plt.ylabel("ICV ratio")
    # # plt.title("Average Travel Time - Timer")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    # plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    # plt.title("平均延迟（s） - 交通灯", fontproperties=zhfont)
    #
    # plt.savefig("baseline_light_time.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    # plt.show()
    # plt.close()
    # # ==============================================
    # plt.figure(8)
    # sns.heatmap(light_wait_lane_data,
    #             cmap=plt.cm.Blues,
    #             fmt='.0f',
    #             vmax=25, vmin=0,
    #             linewidths=0.1,
    #             xticklabels=True, yticklabels=True,
    #             annot=True)
    # # plt.xlabel("Vehicle Generation Probability")
    # # plt.ylabel("ICV ratio")
    # # plt.title("Max Waiting Vehicle on A Lane - Greedy")
    # plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    # plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    # plt.title("单车道最大等待车辆数目（辆）- 交通灯", fontproperties=zhfont)
    #
    # plt.savefig("baseline_light_wait_lane.png", dpi=600, bbox_inches='tight', pad_inches = 0)
    # plt.show()
    # plt.close()