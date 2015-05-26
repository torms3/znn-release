[INPUT]
[INPUT_Conv1a]
[Conv1a]
[Conv1a_Conv1b]
[Conv1b]
[Conv1b_Conv1c]

[INPUT2]
size=1

[INPUT2_Conv1a]
init_type=ReLU
size=3,3,1

[Conv1a]
size=24
activation=relu

[Conv1a_Conv1b]
init_type=ReLU
size=3,3,1

[Conv1b]
size=24
activation=relu

[Conv1b_Conv1c]
init_type=ReLU
size=2,2,1

[Conv1c]
[Conv1c_Conv2a]
[Conv2a]
[Conv2a_Conv2b]
[Conv2b]
[Conv2b_Conv3a]
[Conv3a]
[Conv3a_Conv3b]

[Conv3b]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,2
filter_stride=2,2,1

[Conv3b_Conv4a]
init_type=ReLU
size=3,3,2
load=0

[Conv4a]
size=60
activation=relu
load=0

[Conv4a_Conv4b]
init_type=ReLU
size=3,3,2
load=0

[Conv4b]
size=60
activation=relu
load=0

[Conv4b_Conv4c]
init_type=ReLU
size=3,3,2
load=0

[Conv4c]
size=100
activation=relu
dropout=1
p=0.5
load=0

[Conv4c_OUTPUT]
init_type=ReLU
size=1,1,1
load=0

[OUTPUT]
size=3
activation=forward_logistic
load=0
