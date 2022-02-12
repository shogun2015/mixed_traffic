import math
from enum import Enum
import numpy as np

import traci

from src.const import const_var

# the vehicle number on junction lane
veh_num_in_Junction = {
    ":J_C_0_0": 0,
    ":J_C_1_0": 0,
    ":J_C_2_0": 0,
    ":J_C_3_0": 0,
    ":J_C_4_0": 0,
    ":J_C_5_0": 0,
    ":J_C_6_0": 0,
    ":J_C_7_0": 0
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
        self.stop_vehs = list()  # The veh which is stoping (Receive the stop commend)
        self.veh_speed_junction = list()  # vehicles' speed list in junction

        # For record enter and exit time
        self.travelTimeDict = dict()  # {vehicleID, {enterTime, exitTime}}
        self.in_loop = ['ui_0', 'ui_1', 'di_0', 'di_1', 'ri_0', 'ri_1', 'li_0', 'li_1']
        self.exit_loop = ['uo_0', 'uo_1', 'do_0', 'do_1', 'ro_0', 'ro_1', 'lo_0', 'lo_1']

    def __str__(self):
        return "ICV Controller"

    def feature_step(self, timestep):
        self.time_step = timestep
        self.data_clear_step()
        feature = self._get_vehicles_info()

        avg_speed_junction = 0.
        if self.veh_speed_junction:
            avg_speed_junction = np.mean(self.veh_speed_junction)
        else:
            avg_speed_junction = 10 # Any number that is not 0. 0 leads to simulation reset.

        return feature, avg_speed_junction, self.static_veh_num

    def run_step(self, GAT_actions):
        self.ICV_control(GAT_actions)
        self._collectTraveTime()

    def _get_vehicles_info(self):
        """
        Get vehicles information in Buffered Area
        :return:
        """
        vids = traci.vehicle.getIDList()

        # Produce vehs_in_lane
        for vid in vids:
            veh_color = traci.vehicle.getColor(vid)
            veh_type = "HDV" if (veh_color == (0, 0, 255, 255)) else "ICV"
            veh_pos = self.getVehPos(vid)
            veh_lane = self.getVehLane(vid)  # [laneID, LaneType]
            veh_lane_ID = veh_lane[0]
            veh_lane_Type = veh_lane[1]
            veh_lane_pos = const_var.LANE_LENGTH - 10 - round(traci.vehicle.getLanePosition(vid))

            # Classification by Lane ID
            if veh_lane_ID in const_var.LANE_ID:
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
                veh_num_in_Junction[veh_lane_ID] += 1
                self.veh_speed_junction.append(traci.vehicle.getSpeed(vid))
            else:
                # ignore exit lane
                pass
            # Record static vehicle number
            if traci.vehicle.getSpeed(vid) < 0.1:
                self.static_veh_num += 1

        features = np.zeros(shape=[1, 5], dtype=int)  # a base row for adding more data
        for LaneID in vehs_in_lane.keys():
            # construct ICVs' info in lane (ICV_in_lane)
            for index, veh_info in enumerate(vehs_in_lane[LaneID]):
                if veh_info[1] == "ICV":
                    ICV_in_lane[LaneID].append(
                        [veh_info[0], veh_info[1], veh_info[2], index])  # [vid, ICV/HDV, dist2junction, front_veh_num]
            # construct feature for GAT (features)
            # Produce feature:
            # 1 - Vehicle number in the corresponding junction lane
            feat_veh_num_junction = veh_num_in_Junction[LaneID]
            if ICV_in_lane[LaneID]:
                # 2 - HDV number from the first ICV to stop line
                feat_HDV_num_before = ICV_in_lane[LaneID][0][3]
                # 3 - Distance from the first ICV to stop line
                feat_dist2stop = ICV_in_lane[LaneID][0][2]
                # 4 - Vehicle number after the first ICV
                feat_veh_num_after = len(vehs_in_lane[LaneID]) - ICV_in_lane[LaneID][0][3] - 1
                # 5 - HDV number from the first ICV and the second ICV
                feat_HDV_num_between_ICV = ICV_in_lane[LaneID][1][3] - ICV_in_lane[LaneID][0][3] - 1 if len(
                    ICV_in_lane[LaneID]) > 1 else len(vehs_in_lane[LaneID]) - ICV_in_lane[LaneID][0][3]

            else:
                feat_HDV_num_before = len(vehs_in_lane[LaneID])
                feat_dist2stop = const_var.LANE_LENGTH - 10
                feat_veh_num_after = 0
                feat_HDV_num_between_ICV = 0

            row = np.array([feat_veh_num_junction,
                            feat_HDV_num_before,
                            feat_dist2stop,
                            feat_veh_num_after,
                            feat_HDV_num_between_ICV])
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
            # return [self.junction_sourceLane_map[traci.vehicle.getLaneID(vid)], LaneType.inJunction]
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
                    # print("timestep:{} \t vid:{} \t travel_time:{}".format(self.time_step, veh_id,
                    #                                                        self.travelTimeDict[veh_id].exit_time -
                    #                                                        self.travelTimeDict[veh_id].entry_time))

    def data_clear_step(self):
        self.vehicles_info.clear()
        self.stop_vehs.clear()
        self.veh_speed_junction.clear()
        self.static_veh_num = 0
        for key in veh_num_in_Junction.keys():
            veh_num_in_Junction[key] = 0
        for key in vehs_in_lane.keys():
            vehs_in_lane[key].clear()
        for key in ICV_in_lane.keys():
            ICV_in_lane[key].clear()

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
        self.stop_vehs.append(vehID)
        lane_ID = traci.vehicle.getLaneID(vehID)
        edgeID, laneIndex = lane_ID.spilt('_')
        traci.vehicle.setStop(vehID=vehID, edgeID=edgeID, pos=const_var.LANE_LENGTH - 11., laneIndex=laneIndex)
        # 189. is a magic number. If there is two-lane for junction arm, 11.m prevent vehicle from entering junction

    def ICV_resume(self, vehID):
        if traci.vehicle.getSpeed(vehID) == 0 and vehID in self.stop_vehs:
            traci.vehicle.resume(vehID)
            self.stop_vehs.remove(vehID)
        else:
            traci.vehicle.setSpeed(vehID, -1)

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
