[INPUT]
size=1

[INPUT_C1a]
init_type=Uniform
init_params=0.05
size=3,3,1

[C1a]
size=24
activation=relu

[C1a_C1b]
init_type=Uniform
init_params=0.05
size=3,3,1

[C1b]
size=24
activation=relu

[C1b_C1c]
init_type=normalized
size=2,2,1

[C1c]
size=24
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[C1c_C2a]
init_type=Uniform
init_params=0.05
size=3,3,1

[C2a]
size=36
activation=relu

[C2a_C2b]
init_type=Uniform
init_params=0.05
size=3,3,1

[C2b]
size=36
activation=relu

[C2b_C2c]
init_type=normalized
size=3,3,1

[C2c]
size=36
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=1,1,1

[C2c_C3a]
init_type=Uniform
init_params=0.05
size=3,3,1

[C3a]
size=36
activation=relu

[C3a_C3b]
init_type=normalized
size=3,3,1

[C3b]
size=36
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=1,1,1

[C3b_C4a]
init_type=Uniform
init_params=0.05
size=3,3,1

[C4a]
size=48
activation=relu

[C4a_C4b]
init_type=normalized
size=3,3,1

[C4b]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,2
filter_stride=1,1,1

[C4b_C5a]
init_type=Uniform
init_params=0.05
size=3,3,3

[C5a]
size=60
activation=relu

[C5a_C5b]
init_type=Uniform
init_params=0.05
size=3,3,3

[C5b]
size=60
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,2
filter_stride=1,1,1

[C5b_C6]
init_type=Uniform
init_params=0.05
size=3,3,3

[C6]
size=200
activation=relu

[C6_OUTPUT]
init_type=Uniform
init_params=0.05
size=1,1,1

[OUTPUT]
size=3
activation=forward_logistic