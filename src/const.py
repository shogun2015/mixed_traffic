import numpy as np


class const_var:
    LANE_ID = ["u_0", "u_1", "r_0", "r_1",
               "d_0", "d_1", "l_0", "l_1"]  # up, right, down, left
    JUNCTION_ID = [":J_C_0_0", ":J_C_1_0", ":J_C_2_0", ":J_C_3_0",
                   ":J_C_4_0", ":J_C_5_0", ":J_C_6_0", ":J_C_7_0"]

    LANE_LENGTH = 189.6 # get from traci
    VEH_LENGTH = 5
    MAX_VEH_NUM_Lane = 25    # 25 is the actual num
    MAX_VEH_NUM_Junction = 4    # 25 is the actual num
    STRAIGHT = 30.0
    LEFT_TURN = 48.0
    RIGHT_TURN = 23.0

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
            US  UL  RS  RL  DS  DL  LL  LS
        US  o   o   x   o   o   x   x   x
        UL  o   o   x   x   x   o   x   o
        RS  x   x   o   o   x   o   x   o
        RL  o   x   o   o   x   x   o   x
        DS  o   x   x   x   o   o   o   x
        DL  x   o   o   x   o   o   x   x
        LL  x   x   x   o   o   x   o   o
        LS  x   o   o   x   x   x   o   o
    '''

    lane_adjacent = np.array([[0, 0, 1, 0, 0, 1, 1, 1],
                         [0, 0, 1, 1, 1, 0, 1, 0],
                         [1, 1, 0, 0, 1, 0, 1, 0],
                         [0, 1, 0, 0, 1, 1, 0, 1],
                         [0, 1, 1, 1, 0, 0, 0, 1],
                         [1, 0, 0, 1, 0, 0, 1, 1],
                         [1, 1, 1, 0, 0, 1, 0, 0],
                         [1, 0, 0, 1, 1, 1, 0, 0]], dtype=int)
