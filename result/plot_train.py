from tensorboard.backend.event_processing import event_accumulator
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import interpolate


def smooth(raw_data, weight=0.85):  # weight是平滑度，tensorboard 默认0.6
    # data = pd.read_csv(filepath_or_buffer=csv_path, header=0, names=['Step', 'Value'],
    #                    dtype={'Step': np.int, 'Value': np.float})
    # scalar = data['Value'].values
    last = raw_data[0]
    smoothed = []
    for point in raw_data:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    # save = pd.DataFrame({'Step': data['Step'].values, 'Value': smoothed})
    # save.to_csv('smooth_' + csv_path)
    return smoothed


def expand(raw_data, scale):
    raw_length = len(raw_data)
    result = []
    for index, value in enumerate(raw_data):
        if index == raw_length - 1:
            break
        delta = raw_data[index + 1] - raw_data[index]
        delta_step = delta / scale
        for i in range(scale):
            result.append(raw_data[index] + delta_step * i)
    return result


if __name__ == '__main__':
    zhfont = mpl.font_manager.FontProperties(fname='/home/wuth-3090/.local/share/fonts/SimHei.ttf')

    ea_th_1 = event_accumulator.EventAccumulator(
        "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/output"
        "/03-27/least_reward_ControlInterval_1_ActionGen_thresh/events.out.tfevents.1648393434.wuth3090.650254.0")
    ea_th_5 = event_accumulator.EventAccumulator(
        "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/output/03-27"
        "/least_reward_ControlInterval_5_ActionGen_thresh/events.out.tfevents.1648393445.wuth3090.650311.0")
    ea_th_25 = event_accumulator.EventAccumulator(
        "/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic/output/03-27"
        "/least_reward_ControlInterval_25_ActionGen_thresh/events.out.tfevents.1648393449.wuth3090.650348.0")

    ea_th_1.Reload()
    ea_th_5.Reload()
    ea_th_25.Reload()
    # print(ea_th_1.scalars.Keys())
    # print(ea_th_5.scalars.Keys())
    # print(ea_th_25.scalars.Keys())

    loss_th_1 = ea_th_1.scalars.Items('loss')
    loss_th_5 = ea_th_5.scalars.Items('loss')
    loss_th_25 = ea_th_25.scalars.Items('loss')
    # mean_reward = ea_th_1.scalars.Items("Mean reward at each episode")
    # print("loss length:{}".format(len(loss)))
    # print("mean_reward length:{}".format(len(mean_reward)))

    # print(mean_reward)

    # loss_th_1_index = [i.step for i in loss_th_1]
    # loss_th_5_index = [i.step for i in loss_th_5]
    # loss_th_25_index = [i.step for i in loss_th_25]
    loss_th_1_value = [i.value for i in loss_th_1]
    loss_th_5_value = [i.value for i in loss_th_5]
    loss_th_25_value = [i.value for i in loss_th_25]

    # loss_th_1_value = smooth(loss_th_1_value, 0.999)
    # loss_th_5_value = smooth(loss_th_5_value, 0.999)
    # loss_th_25_value = smooth(loss_th_25_value, 0.999)

    loss_th_1_value_expand = expand(loss_th_1_value, 1)
    loss_th_5_value_expand = expand(loss_th_5_value, 1)
    loss_th_25_value_expand = expand(loss_th_25_value, 1)

    # f_5 = interpolate.interp1d(loss_th_5_index, loss_th_5_value, 'nearest')
    # f_25 = interpolate.interp1d(loss_th_25_index, loss_th_25_value, 'nearest')
    #
    # loss_th_5_value_new = f_5(loss_th_1_index)
    # loss_th_25_value_new = f_25(loss_th_1_index)

    plt.figure(0)
    # plt.plot(loss_th_1_index, loss_th_1_value, label="loss th 1")
    # plt.plot(loss_th_1_index, loss_th_5_value_new, label="loss th 5")
    # plt.plot(loss_th_1_index, loss_th_25_value_new, label="loss th 25")
    plt.plot(loss_th_1_value, label="loss th 1")
    plt.plot(loss_th_5_value_expand, label="loss th 5")
    plt.plot(loss_th_25_value_expand, label="loss th 25")
    plt.xlabel("步数", fontproperties=zhfont)
    plt.ylabel("损失", fontproperties=zhfont)
    plt.title("标题", fontproperties=zhfont)
    plt.legend(loc='upper right')
    plt.savefig("test_train.png", dpi=600, bbox_inches='tight', pad_inches=0)
    plt.grid()
    plt.show()
