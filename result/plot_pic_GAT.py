import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt

if __name__ == '__main__':

    zhfont = mpl.font_manager.FontProperties(fname='/home/wuth-3090/.local/share/fonts/SimHei.ttf')

    # batch_32_pass_rewardx1_GAT_step.csv
    path_base = "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/result/"
    exp_name = "extract_ControlInterval_25_ActionGen_QtoA_mlp_GAT"

    file_time = path_base + "{}_time.csv".format(exp_name)
    file_step = path_base + "{}_step.csv".format(exp_name)
    file_wait_lane = path_base + "{}_wait_lane.csv".format(exp_name)
    np_gat_time = np.loadtxt(open(file_time, "r"), delimiter=",", skiprows=0)
    np_gat_step = np.loadtxt(open(file_step, "r"), delimiter=",", skiprows=0)
    np_gat_wait_lane = np.loadtxt(open(file_wait_lane, "r"), delimiter=",", skiprows=0)

    np_gat_time = np_gat_time - 12
    np_gat_step = (np_gat_step + 1) / 4

    prob = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
    icv_ratio = [0.1, 0.2, 0.3, 0.4, 0.5, 1]
    # X, Y = np.meshgrid(np_prob, np_icv_ratio)
    gat_time = {}
    gat_step = {}
    gat_wait_lane = {}

    for i in range(len(prob)):
        gat_time[prob[i]] = np_gat_time[i]
        gat_step[prob[i]] = np_gat_step[i]
        gat_wait_lane[prob[i]] = np_gat_wait_lane[i]
    gat_time_data = pd.DataFrame(gat_time, index=icv_ratio, columns=prob)
    gat_step_data = pd.DataFrame(gat_step, index=icv_ratio, columns=prob)
    gat_wait_lane_data = pd.DataFrame(gat_wait_lane, index=icv_ratio, columns=prob)

    # ----------------------------------------------------------------------------------

    plt.figure(1)
    sns.heatmap(gat_time_data,
                cmap=plt.cm.Blues,
                fmt='.1f',
                vmax=50, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("平均延迟（s） - GRL-UIV", fontproperties=zhfont)

    png_name = exp_name + "_time.png"
    plt.savefig(png_name, dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
    # ==============================================
    plt.figure(2)
    sns.heatmap(gat_step_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=1000, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("“死锁”发生时间（秒） - GRL-UIV", fontproperties=zhfont)

    png_name = exp_name + "_step.png"
    plt.savefig(png_name, dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
    # ==============================================
    plt.figure(3)
    sns.heatmap(gat_wait_lane_data,
                cmap=plt.cm.Blues,
                fmt='.0f',
                vmax=25, vmin=0,
                linewidths=0.1,
                xticklabels=True, yticklabels=True,
                annot=True)
    plt.xlabel("车辆产生率（辆/秒）", fontproperties=zhfont)
    plt.ylabel("智能网联汽车占比（%）", fontproperties=zhfont)
    plt.title("单车道最大等待车辆数目（辆） - GRL-UIV", fontproperties=zhfont)

    png_name = exp_name + "_wait_lane.png"
    plt.savefig(png_name, dpi=600, bbox_inches='tight', pad_inches = 0)
    plt.show()
    plt.close()
