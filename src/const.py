import numpy as np


class const_var:
    LANE_ID = ["u_0", "u_1", "r_0", "r_1",
               "d_0", "d_1", "l_0", "l_1"]  # up, right, down, left
    JUNCTION_ID = [":J_C_0_0", ":J_C_1_0", ":J_C_2_0", ":J_C_3_0",
                   ":J_C_4_0", ":J_C_5_0", ":J_C_6_0", ":J_C_7_0"]

    action_gen_list = ["thresh", "QtoA"]

    """
    timestep:247 	 vid:DToU_HDV.0 	 travel_time:12.047461378523302
    timestep:900 	 vid:DToU_HDV.1 	 travel_time:13.599140835551509
    timestep:926 	 vid:DToU_HDV.2 	 travel_time:12.621194742722452
    timestep:1133 	 vid:DToU_HDV.3 	 travel_time:11.826522169773199
    timestep:1164 	 vid:DToU_HDV.4 	 travel_time:12.39533961637619
    timestep:1782 	 vid:DToU_HDV.5 	 travel_time:15.383344952756431
    timestep:1941 	 vid:DToU_HDV.6 	 travel_time:12.194406416369702
    timestep:2075 	 vid:DToU_HDV.7 	 travel_time:14.414391752061704
    timestep:2845 	 vid:DToU_HDV.8 	 travel_time:13.602575814198985
    timestep:3130 	 vid:DToU_HDV.9 	 travel_time:14.907828948132533
    
    timestep:252 	 vid:DToL_HDV.0 	 travel_time:13.14479086484345
    timestep:905 	 vid:DToL_HDV.1 	 travel_time:14.683012529755842
    timestep:930 	 vid:DToL_HDV.2 	 travel_time:13.672820769975147
    timestep:1137 	 vid:DToL_HDV.3 	 travel_time:12.85512420491628
    timestep:1169 	 vid:DToL_HDV.4 	 travel_time:13.475640423646809
    timestep:1787 	 vid:DToL_HDV.5 	 travel_time:16.54297467692453
    timestep:1947 	 vid:DToL_HDV.6 	 travel_time:13.343957844318197
    timestep:2081 	 vid:DToL_HDV.7 	 travel_time:15.576137824633918
    timestep:2851 	 vid:DToL_HDV.8 	 travel_time:14.742097966388997
    timestep:3136 	 vid:DToL_HDV.9 	 travel_time:16.087498437312206
    timestep:3819 	 vid:DToL_HDV.10 	 travel_time:12.72306209182375
    timestep:4011 	 vid:DToL_HDV.11 	 travel_time:13.54073123891169
    """

    ThroughTime = { "UToD": 12,
                    "DToU": 12,
                    "LToR": 12,
                    "RToL": 12,
                    "UToR": 12,
                    "DToL": 12,
                    "LToU": 12,
                    "RToD": 12}

    LANE_LENGTH = 189.6  # get from traci
    JUNCTION_LENGTH = 20
    VEH_LENGTH = 5
    MINI_GAP = 2.5
    CELL_SIZE = VEH_LENGTH + MINI_GAP

    # CELL_NUM_JUNCTION = 4     # 20 // 5
    # CELL_NUM_LANE = 38        # 190 // 5
    CELL_NUM_JUNCTION = 3       # 20 // 7.5
    CELL_NUM_LANE = 26          # 190 // 7.5

    STRAIGHT = 30.0
    LEFT_TURN = 48.0
    RIGHT_TURN = 23.0

    # code for vehicle type in cell
    class VehType:
        HDV = 1
        ICV = 2

    junction_sourceLane_map = {
        ":J_C_0_0": "u_0",
        ":J_C_1_0": "u_1",
        ":J_C_2_0": "r_0",
        ":J_C_3_0": "r_1",
        ":J_C_4_0": "d_0",
        ":J_C_5_0": "d_1",
        ":J_C_6_0": "l_0",
        ":J_C_7_0": "l_1"
    }

    sourceLane_junction_map = {
        "u_0": ":J_C_0_0",
        "u_1": ":J_C_1_0",
        "r_0": ":J_C_2_0",
        "r_1": ":J_C_3_0",
        "d_0": ":J_C_4_0",
        "d_1": ":J_C_5_0",
        "l_0": ":J_C_6_0",
        "l_1": ":J_C_7_0"
    }

    '''
        It is a two-lane intersection with left-turning.

            US  UL
                            RS
                            RL
        LL              
        LS              
                    DL  DS
        ====================================    
            US  UL  RS  RL  DS  DL  LS  LL
        US  o   o   x   o   o   x   x   x
        UL  o   o   x   x   x   o   o   x
        RS  x   x   o   o   x   o   o   x
        RL  o   x   o   o   x   x   x   0
        DS  o   x   x   x   o   o   x   o
        DL  x   o   o   x   o   o   x   x
        LS  x   o   o   x   x   x   o   o
        LL  x   x   x   o   o   x   o   o
    '''

    lane_adjacent = np.array([[0, 0, 1, 0, 0, 1, 1, 1],
                              [0, 0, 1, 1, 1, 0, 0, 1],
                              [1, 1, 0, 0, 1, 0, 0, 1],
                              [0, 1, 0, 0, 1, 1, 1, 0],
                              [0, 1, 1, 1, 0, 0, 1, 0],
                              [1, 0, 0, 1, 0, 0, 1, 1],
                              [1, 0, 0, 1, 1, 1, 0, 0],
                              [1, 1, 1, 0, 0, 1, 0, 0]], dtype=int)
