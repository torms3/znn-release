[INPUT]
size=1

[INPUT_C1]
init_type=Uniform
init_params=0.05
size=3,3,1

[C1]
size=24
activation=relu

[C1_C2]
init_type=Uniform
init_params=0.05
size=3,3,1

[C2]
size=24
activation=relu

[C2_C3]
init_type=normalized
size=2,2,1

[C3]
size=24
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C3_C4]
init_type=Uniform
init_params=0.05
size=3,3,1

[C4]
size=36
activation=relu

[C4_C5]
init_type=normalized
size=3,3,1

[C5]
size=36
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C5_C6]
init_type=Uniform
init_params=0.05
size=3,3,1

[C6]
size=48
activation=relu

[C6_C7]
init_type=normalized
size=3,3,1

[C7]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,2
filter_stride=1,1,1

[C7_C8]
init_type=Uniform
init_params=0.05
size=3,3,3

[C8]
size=60
activation=relu

[C8_C9]
init_type=Uniform
init_params=0.05
size=3,3,3

[C9]
size=60
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,2
filter_stride=1,1,1

[C9_FC]
init_type=Uniform
init_params=0.05
size=3,3,3

[FC]
size=200
activation=relu

[FC_OUTPUT]
init_type=Uniform
init_params=0.05
size=1,1,1

[OUTPUT]
size=3
activation=forward_logistic