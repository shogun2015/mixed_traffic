import math
import cv2
from enum import Enum

import traci
from src.const import const_var


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


class BasicController:

    def __init__(self):
        # initial variable
        self.ignore_rightTurn = False
        self.vehicles_info = dict()  # {vehicleID, [laneID, LaneType, LanePos]}
        self.travelTimeDict = dict()  # {vehicleID, {enterTime, exitTime}}
        self.collisionVehPair = []  # [[veh1id, veh2id],...]
        self.time_step = 0
        self.numCollision = -1
        self.in_loop = ['ui_0', 'ui_1', 'di_0', 'di_1', 'ri_0', 'ri_1', 'li_0', 'li_1']
        self.exit_loop = ['uo_0', 'uo_1', 'do_0', 'do_1', 'ro_0', 'ro_1', 'lo_0', 'lo_1']

        # get info from const.py
        self.LANE_ID_set = const_var.LANE_ID
        self.junction_sourceLane_map = const_var.junction_sourceLane_map
        self.Junction_ID_set = const_var.JUNCTION_ID

        # config the controller
        self.setIgnoreTurnRight(True)

    def setIgnoreTurnRight(self, option):
        if option:
            self.ignore_rightTurn = True
            self.LANE_ID_set = list(set(const_var.LANE_ID) - {"l_0", "b_0", "r_0", "u_0"})

    def __str__(self):
        return "Basic Controller"

    def simulation_step(self, time_step):
        '''
        control each vehicle's action at a time step
        :param time_step:
        :return:
        '''
        self.time_step = time_step
        self._get_vehicles_info()
        self._collectCollisionNum()
        self._collectTraveTime()
        self.controlVehicle()

    def _get_vehicles_info(self):
        '''
        Get the vehicles information in Buffered Area
        :return:
        '''
        self.vehicles_info.clear()
        vids = traci.vehicle.getIDList()

        for vid in vids:
            veh_color = traci.vehicle.getColor(vid)
            veh_pos = self.getVehPos(vid)
            veh_lane = self.getVehLane(vid)  # [laneID, LaneType]
            veh_lane_ID = veh_lane[0]
            veh_lane_Type = veh_lane[1]
            veh_lane_pos = self.getVehLanePosition(vid)

            if veh_lane_ID in self.LANE_ID_set:
                maxAllowedSpeed = traci.lane.getMaxSpeed(veh_lane_ID)
                traci.vehicle.setMaxSpeed(vid, maxAllowedSpeed)
                self.vehicles_info[vid] = [veh_lane_ID, veh_lane_Type, veh_lane_pos]

    def getVehPos(self, vid):
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
        if traci.vehicle.getLaneID(vid) in self.LANE_ID_set:
            # vehicle on entering lane
            return traci.vehicle.getLanePosition(vid)
        elif traci.vehicle.getLaneID(vid) in self.Junction_ID_set:
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
        if traci.vehicle.getLaneID(vid) in self.LANE_ID_set:
            # vehicle on entering lane
            return [traci.vehicle.getLaneID(vid), LaneType.Entering]
        elif traci.vehicle.getLaneID(vid) in self.Junction_ID_set:
            # vehicle in merging area
            return [self.junction_sourceLane_map[traci.vehicle.getLaneID(vid)], LaneType.inJunction]
        else:
            # vehicle on exiting lane
            return [traci.vehicle.getLaneID(vid), LaneType.Exiting]

    def _collectCollisionNum(self):
        """
        Record collision vehicle pair
        :return:
        """
        _vids = traci.vehicle.getIDList()
        # O(n2) traverse all pair inside junction
        for vid in _vids:
            veh_lane_type_1 = self.getVehLane(vid)[1]
            if veh_lane_type_1 is not LaneType.inJunction:
                continue
            for vidOther in _vids:
                veh_lane_type_2 = self.getVehLane(vidOther)[1]
                if veh_lane_type_2 is not LaneType.inJunction:
                    continue
                if vid != vidOther:
                    veh1Pos = self.getVehPos(vid)
                    veh2Pos = self.getVehPos(vidOther)
                    width = traci.vehicle.getWidth(vid)
                    height = traci.vehicle.getHeight(vid)
                    angle1 = traci.vehicle.getAngle(vid)
                    angle2 = traci.vehicle.getAngle(vidOther)
                    rect1 = ((veh1Pos[0], veh1Pos[1]), (width, height), angle1)
                    rect2 = ((veh2Pos[0], veh2Pos[1]), (width, height), angle2)

                    if cv2.rotatedRectangleIntersection(rect1, rect2)[0] in [1, 2]:
                        if [vid, vidOther] not in self.collisionVehPair \
                                and [vidOther, vid] not in self.collisionVehPair:
                            self.collisionVehPair.append([vid, vidOther])
        numCollision_inFunction = len(self.collisionVehPair)
        if self.numCollision != numCollision_inFunction:
            print("Timestamp:{} \t The collision num: {}".format(self.time_step, numCollision_inFunction))
            self.numCollision = numCollision_inFunction

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
                    print("timestep:{} \t vid:{} \t travel_time:{}".format(self.time_step, veh_id,
                                                                           self.travelTimeDict[veh_id].exit_time -
                                                                           self.travelTimeDict[veh_id].entry_time))

    def controlVehicle(self):
        """
        Control each vehicle action
        :return:
        """
        # vids = traci.vehicle.getLaneID()
        #
        # for vid in vids:
        pass
