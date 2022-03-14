
class controller_greedy:
    def __init__(self):
        pass

    def __str__(self):
        return "greedy"

    def run_step(self, lane_static_veh_num):
        dir_static_num = []
        dir_static_num.append(lane_static_veh_num[0]+lane_static_veh_num[1])
        dir_static_num.append(lane_static_veh_num[2]+lane_static_veh_num[3])
        dir_static_num.append(lane_static_veh_num[4]+lane_static_veh_num[5])
        dir_static_num.append(lane_static_veh_num[6]+lane_static_veh_num[7])

        index = dir_static_num.index(max(dir_static_num))

        if index == 0:
            return [1, 1, 0, 0, 0, 0, 0, 0]
        elif index == 1:
            return [0, 0, 1, 1, 0, 0, 0, 0]
        elif index == 2:
            return [0, 0, 0, 0, 1, 1, 0, 0]
        else:
            return [0, 0, 0, 0, 0, 0, 1, 1]


