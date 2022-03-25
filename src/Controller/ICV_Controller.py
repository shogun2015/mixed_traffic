import sys

sys.path.append("/home/wuth-3090/Code/yz_mixed_traffic/mixed_traffic/mixed_traffic")
import math
from enum import Enum
import numpy as np

import traci

from src.const import const_var

# the vehicle number on junction lane, using this with junction_sourceLane_map
veh_num_in_Junction = {
    "u_0": 0,
    "u_1": 0,
    "r_0": 0,
    "r_1": 0,
    "d_0": 0,
    "d_1": 0,
    "l_0": 0,
    "l_1": 0
}

# a vehicle info list in dict
# item in vehicle info list [id, type, pos]
vehs_in_lane = {
    "u_0": [],
    "u_1": [],
    "r_0": [],
    "r_1": [],
    "d_0": [],
    "d_1": [],
    "l_0": [],
    "l_1": []
}

vehs_in_junction = {
    ":J_C_0_0": [],
    ":J_C_1_0": [],
    ":J_C_2_0": [],
    ":J_C_3_0": [],
    ":J_C_4_0": [],
    ":J_C_5_0": [],
    ":J_C_6_0": [],
    ":J_C_7_0": []
}

# a dict contains each lane's ICVs
ICV_in_lane = {
    "u_0": [],
    "u_1": [],
    "r_0": [],
    "r_1": [],
    "d_0": [],
    "d_1": [],
    "l_0": [],
    "l_1": []
}
HDV_in_lane = {
    "u_0": [],
    "u_1": [],
    "r_0": [],
    "r_1": [],
    "d_0": [],
    "d_1": [],
    "l_0": [],
    "l_1": []
}


class LaneType(Enum):
    Entering = 1
    inJunction = 2
    Exiting = 3


class TravelTime(object):
    """
    Record Travel Time
    """

    def __init__(self):
        self.entry_time = 0
        self.exit_time = 0

    def add_entry_time(self, time):
        self.entry_time = time

    def add_exit_time(self, time):
        self.exit_time = time


class ICV_Controller:

    def __init__(self):
        # initial variable
        self.time_step = 0
        self.static_veh_num = 0
        self.vehicles_info = dict()  # {vehicleID, [laneID, LaneType, LanePos]}
        # self.stop_vehs = list()  # The veh which is stoping (Receive the stop commend)
        self.veh_speed_junction = dict()  # vehicles' speed list in junction

        # For record enter and exit time
        self.travelTimeDict = dict()  # {vehicleID, {enterTime, exitTime}}
        self.in_loop = ['ui_0', 'ui_1', 'di_0', 'di_1', 'ri_0', 'ri_1', 'li_0', 'li_1']
        self.exit_loop = ['uo_0', 'uo_1', 'do_0', 'do_1', 'ro_0', 'ro_1', 'lo_0', 'lo_1']
        self.lane_vehicle_num = [0 for _ in range(8)]

        self.travel_time = dict()

    def __str__(self):
        return "ICV Controller"

    def feature_step(self, timestep):
        self.time_step = timestep
        self.data_clear_step()
        feature = self._get_vehicles_info()

        exit_vehicle_num = self.collect_exit_vehicle_num()
        self._collectTraveTime()
        """
        feature - 特征矩阵
        self.static_veh_num - 静止车数目
        is_all_static_junction - 路口内是否全为静止车
        min(self.lane_static_veh_num) - 静止车辆最多车道的车辆数
        max(self.lane_static_veh_num) - 静止车辆最少车道的车辆数
        """
        return feature, self.static_veh_num, self.lane_static_veh_num, exit_vehicle_num, self.travel_time

    def run_step(self, GAT_actions):
        self.ICV_control(GAT_actions)
        self._collectTraveTime()

    def _get_vehicles_info(self):
        """
        Get vehicles information in Buffered Area
        :return:
        """
        vids = traci.vehicle.getIDList()
        self.lane_static_veh_num = [0 for _ in range(8)]
        # Produce vehs_in_lane
        for vid in vids:
            veh_color = traci.vehicle.getColor(vid)
            veh_type = "HDV" if (veh_color == (0, 0, 255, 255)) else "ICV"
            veh_pos = self.getVehPos(vid)
            veh_lane = self.getVehLane(vid)  # [laneID, LaneType]
            veh_lane_ID = veh_lane[0]
            veh_lane_Type = veh_lane[1]

            veh_length = traci.vehicle.getLength(vid)

            # Classification by Lane ID
            if veh_lane_ID in const_var.LANE_ID:
                veh_lane_pos = const_var.LANE_LENGTH - round(traci.vehicle.getLanePosition(vid))
                if vehs_in_lane[veh_lane_ID]:  # vehicle list not empty
                    for index, value in enumerate(vehs_in_lane[veh_lane_ID]):
                        if veh_lane_pos < value[2]:  # the shortest dist to junction first
                            vehs_in_lane[veh_lane_ID].insert(index, [vid, veh_type, veh_lane_pos])
                            break
                        elif index + 1 == len(vehs_in_lane[veh_lane_ID]):
                            vehs_in_lane[veh_lane_ID].insert(index + 1, [vid, veh_type, veh_lane_pos])
                            break
                else:
                    vehs_in_lane[veh_lane_ID].insert(0, [vid, veh_type, veh_lane_pos])
            elif veh_lane_ID in const_var.JUNCTION_ID:
                veh_lane_pos = const_var.JUNCTION_LENGTH - round(traci.vehicle.getLanePosition(vid))
                if vehs_in_junction[veh_lane_ID]:
                    for index, value in enumerate(vehs_in_junction[veh_lane_ID]):
                        if veh_lane_pos < value[2]:  # the shortest dist to junction first
                            vehs_in_junction[veh_lane_ID].insert(index, [vid, veh_type, veh_lane_pos])
                            break
                        elif index + 1 == len(vehs_in_junction[veh_lane_ID]):
                            vehs_in_junction[veh_lane_ID].insert(index + 1, [vid, veh_type, veh_lane_pos])
                            break
                else:
                    vehs_in_junction[veh_lane_ID].insert(0, [vid, veh_type, veh_lane_pos])
                self.veh_speed_junction[vid] = traci.vehicle.getSpeed(vid)
            else:
                # ignore exit lane
                pass
            # Record static vehicle number
            if traci.vehicle.getSpeed(vid) < 0.1:
                if veh_lane_ID in const_var.LANE_ID:
                    self.lane_static_veh_num[const_var.LANE_ID.index(veh_lane_ID)] += 1
                self.static_veh_num += 1

        # a base row for adding more data
        features = np.append(np.zeros(const_var.CELL_NUM_JUNCTION + const_var.CELL_NUM_LANE, dtype=int),
                             np.zeros(const_var.CELL_NUM_JUNCTION + const_var.CELL_NUM_LANE, dtype=float))
        # features = np.append(np.zeros(const_var.CELL_NUM_LANE, dtype=int), np.zeros(const_var.CELL_NUM_LANE, dtype=float))

        for LaneID in vehs_in_lane.keys():
            # construct ICVs' info in lane (ICV_in_lane)
            for index, veh_info in enumerate(vehs_in_lane[LaneID]):

                # check if ICV can stop before stopline
                speed = traci.vehicle.getSpeed(veh_info[0])
                maxDecel = traci.vehicle.getDecel(veh_info[0])  # the maximal comfortable deceleration in m/s^2
                dist_require = (-speed * speed) / (-2 * maxDecel)  # maxDecel is a positive number
                dist2stop = veh_info[2]
                stoppable = True if dist2stop > dist_require else False

                max_spped = traci.vehicle.getMaxSpeed(veh_info[0])

                if veh_info[1] == "ICV" and stoppable:
                    ICV_in_lane[LaneID].append(
                        [veh_info[0], "ICV", veh_info[2], index])  # [vid, ICV/HDV, dist2junction, front_veh_num]
                else:
                    HDV_in_lane[LaneID].append(
                        [veh_info[0], "HDV", veh_info[2], index])  # [vid, ICV/HDV, dist2junction, front_veh_num]

            # Produce feature:
            """
            # 1 - Vehicle number in the corresponding junction lane
            feat_veh_num_junction = veh_num_in_Junction[LaneID]
            if ICV_in_lane[LaneID]:
                # 2 - HDV number from the first ICV to stop line
                feat_HDV_num_before = ICV_in_lane[LaneID][0][3]
                if not self.can_stop(ICV_in_lane[LaneID][0][0]):
                    feat_HDV_num_before += 1
                # 3 - Distance from the first ICV to stop line
                feat_dist2stop = ICV_in_lane[LaneID][0][2]
                # 4 - Vehicle number after the first ICV
                feat_veh_num_after = len(vehs_in_lane[LaneID]) - ICV_in_lane[LaneID][0][3] - 1
                # 4.5 - distance between 1st and 2nd ICV
                if len(ICV_in_lane[LaneID]) > 1:
                    feat_dist_between_ICV = ICV_in_lane[LaneID][1][2] - ICV_in_lane[LaneID][0][2]
                else:
                    feat_dist_between_ICV = const_var.LANE_LENGTH - feat_dist2stop
                # 5 - HDV number from the first ICV and the second ICV
                feat_HDV_num_between_ICV = ICV_in_lane[LaneID][1][3] - ICV_in_lane[LaneID][0][3] - 1 if len(
                    ICV_in_lane[LaneID]) > 1 else len(vehs_in_lane[LaneID]) - ICV_in_lane[LaneID][0][3]
            else:
                feat_HDV_num_before = len(vehs_in_lane[LaneID])
                feat_dist2stop = const_var.LANE_LENGTH
                feat_veh_num_after = 0
                feat_dist_between_ICV = 0
                feat_HDV_num_between_ICV = 0

            # feature normalize
            feat_veh_num_junction_norm = feat_veh_num_junction / const_var.MAX_VEH_NUM_Junction
            feat_HDV_num_before_norm = feat_HDV_num_before / const_var.MAX_VEH_NUM_Lane
            feat_dist2stop_norm = feat_dist2stop / const_var.LANE_LENGTH
            feat_veh_num_after_norm = feat_veh_num_after/ const_var.MAX_VEH_NUM_Lane
            feat_dist_between_ICV_norm = feat_dist_between_ICV / const_var.LANE_LENGTH
            feat_HDV_num_between_ICV_norm = feat_HDV_num_between_ICV / const_var.MAX_VEH_NUM_Lane

            row = np.array([feat_veh_num_junction_norm,
                            feat_HDV_num_before_norm,
                            feat_dist2stop_norm,
                            feat_veh_num_after_norm,
                            feat_dist_between_ICV_norm,
                            feat_HDV_num_between_ICV_norm])
            """
            cell_array = -1 * np.ones(const_var.CELL_NUM_JUNCTION + const_var.CELL_NUM_LANE, dtype=int)
            cell_speed_array = -1 * np.ones(const_var.CELL_NUM_JUNCTION + const_var.CELL_NUM_LANE, dtype=float)
            # cell_array = -1 * np.ones(const_var.CELL_NUM_LANE, dtype=int)
            # cell_speed_array = -1 * np.ones(const_var.CELL_NUM_LANE, dtype=float)
            junction_lane_ID = const_var.sourceLane_junction_map[LaneID]
            for index, veh_info in enumerate(vehs_in_junction[junction_lane_ID]):
                cell_index = int(veh_info[2] // const_var.CELL_SIZE)
                speed = traci.vehicle.getSpeed(veh_info[0])
                cell_array[cell_index] = const_var.VehType.HDV
                cell_speed_array[cell_index] = speed / 16

            for index, veh_info in enumerate(ICV_in_lane[LaneID]):
                cell_index = int(veh_info[2] // const_var.CELL_SIZE)
                speed = traci.vehicle.getSpeed(veh_info[0])
                cell_array[const_var.CELL_NUM_JUNCTION + cell_index] = const_var.VehType.ICV
                cell_speed_array[const_var.CELL_NUM_JUNCTION + cell_index] = speed / 16
                # cell_array[cell_index] = const_var.VehType.ICV
                # cell_speed_array[cell_index] = speed / 16

            for index, veh_info in enumerate(HDV_in_lane[LaneID]):
                cell_index = int(veh_info[2] // const_var.CELL_SIZE)
                speed = traci.vehicle.getSpeed(veh_info[0])
                cell_array[const_var.CELL_NUM_JUNCTION + cell_index] = const_var.VehType.HDV
                cell_speed_array[const_var.CELL_NUM_JUNCTION + cell_index] = speed / 16
                # cell_array[cell_index] = const_var.VehType.HDV
                # cell_speed_array[cell_index] = speed / 16

            row = np.append(cell_array, cell_speed_array)
            features = np.row_stack([features, row])
        features = features[1:, :]  # remove the first row
        return features

    def getVehPos(self, vid):
        """
        Get vehicle vid's position
        :param vid:
        :return:
        """
        L = traci.vehicle.getLength(vid)
        heading = traci.vehicle.getAngle(vid)
        pos = traci.vehicle.getPosition(vid)
        # print(heading)
        # print(pos)
        # The centre of vehicle
        _x = pos[0] - math.sin((heading / 180) * math.pi) * (L / 2)
        _y = pos[1] - math.cos((heading / 180) * math.pi) * (L / 2)
        return [_x, _y]

    def getVehLanePosition(self, vid):
        """
        Get vid's position on the corresponding lane.
        :param vid:
        :return:
        """
        if traci.vehicle.getLaneID(vid) in const_var.LANE_ID:
            # vehicle on entering lane
            return traci.vehicle.getLanePosition(vid)
        elif traci.vehicle.getLaneID(vid) in const_var.JUNCTION_ID:
            # vehicle in merging area
            return 200 + traci.vehicle.getLanePosition(vid)
        else:
            # vehicle on exiting lane
            return traci.vehicle.getLanePosition(vid)

    def getVehLane(self, vid):
        """
        Get LaneID and LaneType of Vehicle vid
        :param vid: the ID of vehicle
        :return: [LaneID, LaneType]
        """
        if traci.vehicle.getLaneID(vid) in const_var.LANE_ID:
            # vehicle on entering lane
            return [traci.vehicle.getLaneID(vid), LaneType.Entering]
        elif traci.vehicle.getLaneID(vid) in const_var.JUNCTION_ID:
            # vehicle in merging area
            return [traci.vehicle.getLaneID(vid), LaneType.inJunction]
        else:
            # vehicle on exiting lane
            return [traci.vehicle.getLaneID(vid), LaneType.Exiting]

    def _collectTraveTime(self):
        """
        Collect each vehicle traveling time from induction loop
        :return: travel time list for all vehicles
        """
        for in_loop_id in self.in_loop:
            # [(veh_id, veh_length, entry_time, exit_time, vType), ...]
            for vehData in traci.inductionloop.getVehicleData(in_loop_id):
                veh_id = vehData[0]
                entry_time = vehData[2]
                if entry_time != -1.0:
                    self.travelTimeDict[veh_id] = TravelTime()
                    self.travelTimeDict[veh_id].add_entry_time(entry_time)
                    # print("timestep:{} \t vid:{} \t entry:{} ".format(self.time_step, veh_id, entry_time))
        for out_loop_id in self.exit_loop:
            # [(veh_id, veh_length, entry_time, exit_time, vType), ...]
            for vehData in traci.inductionloop.getVehicleData(out_loop_id):
                veh_id = vehData[0]
                exit_time = vehData[3]
                if exit_time != -1.0 and veh_id in list(self.travelTimeDict.keys()):
                    self.travelTimeDict[veh_id].add_exit_time(exit_time)
                    self.travel_time[veh_id] = self.travelTimeDict[veh_id].exit_time \
                                               - self.travelTimeDict[veh_id].entry_time

    def collect_exit_vehicle_num(self):
        exit_vehicle_num = 0
        for out_loop_id in self.exit_loop:
            exit_vehicle_num += len(traci.inductionloop.getVehicleData(out_loop_id))
        return exit_vehicle_num

    def data_clear_step(self):
        self.vehicles_info.clear()
        # self.stop_vehs.clear()
        self.veh_speed_junction.clear()
        self.static_veh_num = 0
        for key in veh_num_in_Junction.keys():
            veh_num_in_Junction[key] = 0
        for key in vehs_in_lane.keys():
            vehs_in_lane[key].clear()
        for key in vehs_in_junction.keys():
            vehs_in_junction[key].clear()
        for key in ICV_in_lane.keys():
            ICV_in_lane[key].clear()
        for key in HDV_in_lane.keys():
            HDV_in_lane[key].clear()

    def ICV_control(self, actions_NN: list):
        for index, laneID in enumerate(ICV_in_lane.keys()):
            if ICV_in_lane[laneID]:
                first_ICV = ICV_in_lane[laneID][0]  # [vid, ICV/HDV, dist2junction, front_veh_num]
                first_ICV_ID = first_ICV[0]
                if actions_NN[index] == 0:
                    self.ICV_stop(first_ICV_ID)
                else:
                    self.ICV_resume(first_ICV_ID)

    def ICV_stop(self, vehID):
        # self.stop_vehs.append(vehID) # put into the buffer stop
        lane_ID = traci.vehicle.getLaneID(vehID)
        edgeID, laneIndex = lane_ID.split("_")
        # change the method of stopping
        self.buffer_stop(vid=vehID, edgeID=edgeID, pos=const_var.LANE_LENGTH - 2., laneIndex=laneIndex)
        # traci.vehicle.setStop(vehID=vehID, edgeID=edgeID, pos=const_var.LANE_LENGTH - 11., laneIndex=laneIndex)
        # If there is two-lane for junction arm, 1.m prevent vehicle from entering junction

    def ICV_resume(self, vehID):
        stopData = traci.vehicle.getStopState(vehID)
        # print("ICV_resume-{}:{}".format(vehID, bin(stopData)))
        # speed = traci.vehicle.getSpeed(vehID)
        # if speed < 0.1 and vehID in self.stop_vehs:
        if stopData & 1 > 0:    # 判断编制为最后1位为1,即为停止状态
            traci.vehicle.resume(vehID)
            # self.stop_vehs.remove(vehID)
        else:
            traci.vehicle.setSpeed(vehID, -1)

    def can_stop(self, vid):
        turnVelocity = 0
        nowSpeed = traci.vehicle.getSpeed(vid)
        lanePostion = traci.vehicle.getLanePosition(vid)
        remainder_min = np.abs((nowSpeed * nowSpeed - turnVelocity * turnVelocity) / (2 * traci.vehicle.getDecel(vid)))
        remainder = 188.6 - lanePostion
        if remainder_min < remainder:
            return True
        else:
            return False

    def buffer_stop(self, vid, edgeID, pos, laneIndex):
        turnVelocity = 0
        nowSpeed = traci.vehicle.getSpeed(vid)
        lanePostion = traci.vehicle.getLanePosition(vid)
        remainder_min = np.abs((nowSpeed * nowSpeed - turnVelocity * turnVelocity) / (2 * traci.vehicle.getDecel(vid)))
        remainder = pos - lanePostion
        if remainder_min < remainder:
            expectedA = np.abs((nowSpeed * nowSpeed - turnVelocity * turnVelocity) / (2 * remainder))
            time_interval = np.abs(nowSpeed - turnVelocity) / expectedA
            # traci.vehicle.slowDown(vid, turnVelocity, time_interval)
            try:
                traci.vehicle.setStop(vid, edgeID, pos, laneIndex=laneIndex)
                # self.stop_vehs.append(vid)
            except:
                pass
        else:
            pass

        # def test_ICV_stop(self):
    #     for key in ICV_in_lane.keys():
    #         if ICV_in_lane[key]:
    #             ICV_info = ICV_in_lane[key][0]  # The first ICV in current Lane
    #             ICV_id = ICV_info[0]
    #             edge_id, lane_id = key.split('_')
    #             # traci.vehicle.changeTarget(vehID=v_id, edgeID=edge_id)
    #             self.ICV_stop(vehID=ICV_id)

    # def test_ICV_resume(self):
    #     for key in ICV_in_lane.keys():
    #         if ICV_in_lane[key]:
    #             ICV_info = ICV_in_lane[key][0]  # The first ICV in current Lane
    #             ICV_id = ICV_info[0]
    #             self.ICV_resume(ICV_id)
