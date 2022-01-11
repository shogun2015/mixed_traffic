import math
from enum import Enum

import traci
from src.const import const_var


class LaneType(Enum):
    Entering = 1
    inJunction = 2
    Exiting = 3

class BasicController:

    def __init__(self):
        # initial variable
        self.ignore_rightTurn = False
        self.vehicles_info = dict() # {vehicleID, [laneID, LaneType, LanePos]}

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

    # control each vehicle's action at a time step
    def simulation_step(self):
        self._get_vehicles_info()

    """
    Get the vehicles information in Buffered Area
    """

    def _get_vehicles_info(self):

        self.vehicles_info.clear()
        vids = traci.vehicle.getIDList()

        for vid in vids:
            veh_color = traci.vehicle.getColor(vid)
            veh_pos = self.getVehPos(vid)
            veh_lane = self.getVehLane(vid) # [laneID, LaneType]
            veh_lane_ID = veh_lane[0]
            veh_lane_Type = veh_lane[1]
            veh_lane_pos = self.getVehLanePosition(vid)

            if veh_lane_ID in self.LANE_ID_set:
                maxAllowedSpeed = traci.lane.getMaxSpeed(veh_lane)
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

    """
    Get LaneID and LaneType of Vehicle vid
    Return [LaneID, LaneType]
    """

    def getVehLane(self, vid):
        if traci.vehicle.getLaneID(vid) in self.LANE_ID_set:
            # vehicle on entering lane
            return [traci.vehicle.getLaneID(vid), LaneType.Entering]
        elif traci.vehicle.getLaneID(vid) in self.Junction_ID_set:
            # vehicle in merging area
            return [self.junction_sourceLane_map[traci.vehicle.getLaneID(vid)], LaneType.inJunction]
        else:
            # vehicle on exiting lane
            return [traci.vehicle.getLaneID(vid), LaneType.Exiting]
