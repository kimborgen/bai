"""
        +-------+-------------------------------+------------------------------------------------+-------------+
        | Index | Descriptions                  | Details                                        | Num Actions |
        +-------+-------------------------------+------------------------------------------------+-------------+
        | 0     | Forward and backward          | 0: noop, 1: forward, 2: back                   | 3           |
        | 1     | Move left and right           | 0: noop, 1: move left, 2: move right            | 3           |
        | 2     | Jump, sneak, and sprint       | 0: noop, 1: jump, 2: sneak, 3:sprint            | 4           |
        | 3     | Camera delta pitch            | 0: -180 degree, 24: 180 degree                  | 25          |
        | 4     | Camera delta yaw              | 0: -180 degree, 24: 180 degree                  | 25          |
        | 5     | Functional actions            | 0: noop, 1: use, 2: drop, 3: attack, 4: craft... | 8           |
        | 6     | Argument for "craft"          | All possible items to be crafted                | 244         |
        | 7     | Argument for "equip", ...     | Inventory slot indices                          | 3           |
        +-------+-------------------------------+------------------------------------------------+-------------+
        0[all], 1[all], 2[all], 3[all], 4[all]
        3+3+4+25+25= 60 possible outputs
        """


        """
 walking speed in mc is 4.317 blocks/s. We want to observe the last 30 seconds * 60fps. Max blocks is therefor 129.51 blocks. 
 Thus if the agents has only traversed for example 10% of that in 30 seconds it is stuck. 10% = 13 blocks ish
"""