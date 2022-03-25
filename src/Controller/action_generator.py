import const
import numpy as np


def Q_to_Action(Q_list):
    size = len(Q_list)
    np_action_list = np.array([-1 for _ in range(size)])
    np_dir_ToDo = np.array([1 for _ in range(size)])

    while -1 in np_action_list:
        Q_list_temp = Q_list * np_dir_ToDo
        Q_max_index = Q_list_temp.argmax()
        np_action_list[Q_max_index] = 1
        np_dir_ToDo[Q_max_index] = 0

        dir_stop = const.const_var.lane_adjacent[Q_max_index]
        for index in range(size):
            if np_action_list[index] == -1 and dir_stop[index] == 1:
                np_action_list[index] = 0
                np_dir_ToDo[index] = 0

    return np_action_list


if __name__ == '__main__':
    res = Q_to_Action([0.1, 0.2, 0.3, 0.4, 0.8, 0.6, 0.7, 0.8])
    print(res)
