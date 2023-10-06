# good protocal
sweep_parameter_prototype = [
    {
        "duration": 10/time_scale_factor , "phi_2_dcx": 3/time_scale_factor, "name": "mix in y direction"
    },

    {"duration": 18/time_scale_factor, "M_12": -0.9, "name": "conditional tilt"}, # conditional tilt

    {"duration": 16/time_scale_factor, "phi_2_dcx": 0, "name": "raise the barrier"}, # raise the barrier

    {"duration": 12/time_scale_factor, "phi_1_dcx": 3/time_scale_factor, "name": "mix in x direction"}, # mix in x direction

    {"duration": 50/time_scale_factor, "phi_1_dcx": 0, "M_12": 0, "name": "4 well potential "}, # 4 well potential
]





{"duration": 100/time_scale_factor, "name": "fix"},
{"duration": 100/time_scale_factor, "phi_1_dcx": 0, "M_12": 0, "name": "4 well potential "}, # 4 well potential
{"duration": 20/time_scale_factor, "name": "4 well potential (constant)"},
