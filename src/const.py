class const_var:
    LANE_ID = ["l_0", "b_0", "r_0", "u_0",
               "l_1", "b_1", "r_1", "u_1",
               "l_2", "b_2", "r_2", "u_2"]  # left, below, right, up
    JUNCTION_ID = [":gneJ11_0_0", ":gneJ11_1_0", ":gneJ11_2_0",
                   ":gneJ11_3_0", ":gneJ11_4_0", ":gneJ11_5_0",
                   ":gneJ11_6_0", ":gneJ11_7_0", ":gneJ11_8_0",
                   ":gneJ11_9_0", ":gneJ11_10_0", ":gneJ11_11_0"]
    LANE_LENGTH = 200
    STRAIGHT = 30.0
    LEFT_TURN = 48.0
    RIGHT_TURN = 23.0

    collisionMatrix = {
        "l_1": {"u_1": 9.2, "b_1": 19.8, "b_2": 12.2, "r_2": 16.8},
        "l_2": {"b_2": 6.5, "u_1": 10.1, "u_2": 18.5, "r_1": 14.9},
        "u_1": {"r_1": 9.2, "l_2": 12.2, "b_2": 16.8, "l_1": 19.8},
        "u_2": {"l_2": 6.5, "r_1": 10.1, "b_1": 14.9, "r_2": 18.5},
        "r_1": {"b_1": 9.2, "u_2": 12.2, "l_2": 16.8, "u_1": 19.8},
        "r_2": {"u_2": 6.5, "b_1": 10.1, "l_1": 14.9, "b_2": 18.5},
        "b_1": {"l_1": 9.2, "r_2": 12.2, "u_2": 16.8, "r_1": 19.8},
        "b_2": {"r_2": 6.5, "l_1": 10.1, "u_1": 14.9, "l_2": 18.5}
    }

    maxTimeMatrix = {
        "l_1": {"u_1": 9.2, "b_1": 19.8, "b_2": 12.2, "r_2": 16.8},
        "l_2": {"b_2": 6.5, "u_1": 10.1, "u_2": 18.5, "r_1": 14.9},
        "u_1": {"r_1": 9.2, "l_2": 12.2, "b_2": 16.8, "l_1": 19.8},
        "u_2": {"l_2": 6.5, "r_1": 10.1, "b_1": 14.9, "r_2": 18.5},
        "r_1": {"b_1": 9.2, "u_2": 12.2, "l_2": 16.8, "u_1": 19.8},
        "r_2": {"u_2": 6.5, "b_1": 10.1, "l_1": 14.9, "b_2": 18.5},
        "b_1": {"l_1": 9.2, "r_2": 12.2, "u_2": 16.8, "r_1": 19.8},
        "b_2": {"r_2": 6.5, "l_1": 10.1, "u_1": 14.9, "l_2": 18.5}
    }

    junction_sourceLane_map = {
        ":gneJ11_0_0": "u_0",
        ":gneJ11_1_0": "u_1",
        ":gneJ11_2_0": "u_2",
        ":gneJ11_3_0": "r_0",
        ":gneJ11_4_0": "r_1",
        ":gneJ11_5_0": "r_2",
        ":gneJ11_6_0": "b_0",
        ":gneJ11_7_0": "b_1",
        ":gneJ11_8_0": "b_2",
        ":gneJ11_9_0": "l_0",
        ":gneJ11_10_0": "l_1",
        ":gneJ11_11_0": "l_2"
    }
