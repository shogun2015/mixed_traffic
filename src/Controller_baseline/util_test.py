import traci
import numpy as np

import const
from src.const import const_var

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
first_ICV_ID_in_lane = {
    "u_0": "",
    "u_1": "",
    "r_0": "",
    "r_1": "",
    "d_0": "",
    "d_1": "",
    "l_0": "",
    "l_1": ""
}


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


class util_test:

    def __init__(self):
        # For record enter and exit time
        self.travelTimeDict = dict()  # {vehicleID, {enterTime, exitTime}}
        self.in_loop = ['ui_0', 'ui_1', 'di_0', 'di_1', 'ri_0', 'ri_1', 'li_0', 'li_1']
        self.exit_loop = ['uo_0', 'uo_1', 'do_0', 'do_1', 'ro_0', 'ro_1', 'lo_0', 'lo_1']
        self.lane_vehicle_num = [0 for _ in range(8)]
        self.travel_time = dict()
        self.veh_speed_junction = list()

    def feature_step(self):
        self.data_clear_step()
        self.get_vehicle_info()

        first_ICVs = []
        for laneID in first_ICV_ID_in_lane.keys():
            first_ICVs.append(first_ICV_ID_in_lane[laneID])

        self.collectTraveTime()

        return first_ICVs, self.travel_time, self.lane_static_veh_num

    def data_clear_step(self):
        self.veh_speed_junction.clear()
        for key in vehs_in_lane.keys():
            vehs_in_lane[key].clear()
        for key in first_ICV_ID_in_lane.keys():
            first_ICV_ID_in_lane[key] = ""

    def get_vehicle_info(self):

        self.lane_static_veh_num = [0 for _ in range(8)]

        vids = traci.vehicle.getIDList()

        # Get vehicle info in each lane (vehs_in_lane[veh_lane_ID])
        for vid in vids:
            veh_color = traci.vehicle.getColor(vid)
            veh_type = "HDV" if (veh_color == (0, 0, 255, 255)) else "ICV"
            veh_lane_ID = traci.vehicle.getLaneID(vid)

            # filter vehicle in entering lane
            if veh_lane_ID in const_var.LANE_ID:
                veh_lane_pos = const_var.LANE_LENGTH - round(traci.vehicle.getLanePosition(vid))
                if vehs_in_lane[veh_lane_ID]:
                    for index, value in enumerate(vehs_in_lane[veh_lane_ID]):
                        if veh_lane_pos < value[2]:  # the shortest dist to junction first
                            vehs_in_lane[veh_lane_ID].insert(index, [vid, veh_type, veh_lane_pos])
                            break
                        elif index + 1 == len(vehs_in_lane[veh_lane_ID]):
                            vehs_in_lane[veh_lane_ID].insert(index + 1, [vid, veh_type, veh_lane_pos])
                            break
                else:
                    vehs_in_lane[veh_lane_ID].insert(0, [vid, veh_type, veh_lane_pos])

            if veh_lane_ID in const_var.JUNCTION_ID:
                self.veh_speed_junction.append(traci.vehicle.getSpeed(vid))

            # Record static vehicle number
            if traci.vehicle.getSpeed(vid) < 0.1:
                if veh_lane_ID in const_var.LANE_ID:
                    self.lane_static_veh_num[const_var.LANE_ID.index(veh_lane_ID)] += 1

        for LaneID in vehs_in_lane.keys():
            is_update = False
            for index, veh_info in enumerate(vehs_in_lane[LaneID]):
                # # check if ICV can stop before stopline
                # speed = traci.vehicle.getSpeed(veh_info[0])
                # maxDecel = traci.vehicle.getDecel(veh_info[0])  # the maximal comfortable deceleration in m/s^2
                # dist_require = (-speed * speed) / (-2 * maxDecel)  # maxDecel is a positive number
                # dist2stop = veh_info[2]
                # stoppable = True if dist2stop > dist_require else False

                # if veh_info[1] == "ICV" and stoppable:
                if veh_info[1] == "ICV":
                    first_ICV_ID_in_lane[LaneID] = veh_info[0]
                    is_update = True
                    break
            if is_update:
                continue

    def ICV_control(self, actions_NN: list):
        for index, laneID in enumerate(first_ICV_ID_in_lane.keys()):
            if first_ICV_ID_in_lane[laneID] != "":
                if actions_NN[index] == 0:
                    self.ICV_stop(first_ICV_ID_in_lane[laneID])
                else:
                    self.ICV_resume(first_ICV_ID_in_lane[laneID])

    def ICV_stop(self, vehID):
        # self.stop_vehs.append(vehID) # put into the buffer stop
        lane_ID = traci.vehicle.getLaneID(vehID)
        edgeID, laneIndex = lane_ID.split("_")
        # change the method of stopping
        self.buffer_stop(vid=vehID, edgeID=edgeID, pos=const_var.LANE_LENGTH - 2., laneIndex=laneIndex)

    def ICV_resume(self, vehID):
        stopData = traci.vehicle.getStopState(vehID)
        if stopData & 1 > 0:  # 判断编制为最后1位为1,即为停止状态
            traci.vehicle.resume(vehID)
        else:
            traci.vehicle.setSpeed(vehID, -1)

    def can_stop(self, vid):
        turnVelocity = 0
        nowSpeed = traci.vehicle.getSpeed(vid)
        lanePostion = traci.vehicle.getLanePosition(vid)
        remainder_min = np.abs(
            (nowSpeed * nowSpeed - turnVelocity * turnVelocity) / (2 * traci.vehicle.getDecel(vid)))
        remainder = const_var.LANE_LENGTH - lanePostion
        if remainder_min < remainder:
            return True
        else:
            return False

    def buffer_stop(self, vid, edgeID, pos, laneIndex):
        turnVelocity = 0
        nowSpeed = traci.vehicle.getSpeed(vid)
        lanePostion = traci.vehicle.getLanePosition(vid)
        remainder_min = np.abs(
            (nowSpeed * nowSpeed - turnVelocity * turnVelocity) / (2 * traci.vehicle.getDecel(vid)))
        remainder = pos - lanePostion
        if remainder_min < remainder:
            expectedA = np.abs((nowSpeed * nowSpeed - turnVelocity * turnVelocity) / (2 * remainder))
            time_interval = np.abs(nowSpeed - turnVelocity) / expectedA
            try:
                traci.vehicle.setStop(vid, edgeID, pos, laneIndex=laneIndex)
            except:
                pass
        else:
            pass

    def collectTraveTime(self):
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
                    # print("vid:{} \t entry:{} ".format(veh_id, entry_time))
        for out_loop_id in self.exit_loop:
            # [(veh_id, veh_length, entry_time, exit_time, vType), ...]
            for vehData in traci.inductionloop.getVehicleData(out_loop_id):
                veh_id = vehData[0]
                exit_time = vehData[3]
                if exit_time != -1.0 and veh_id in list(self.travelTimeDict.keys()):
                    self.travelTimeDict[veh_id].add_exit_time(exit_time)
                    self.travel_time[veh_id] = self.travelTimeDict[veh_id].exit_time \
                                               - self.travelTimeDict[veh_id].entry_time
                    # print("vid:{} \t travel_time:{}".format(veh_id,
                    #                                         self.travelTimeDict[veh_id].exit_time -
                    #                                         self.travelTimeDict[veh_id].entry_time))
