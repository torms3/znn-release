[INPUT]
size=1

[INPUT_Conv1a]
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
size=24
activation=relu
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[Conv1c_Conv2a]
init_type=ReLU
size=3,3,1

[Conv2a]
size=36
activation=relu

[Conv2a_Conv2b]
init_type=ReLU
size=3,3,1

[Conv2b]
size=36
activation=relu
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[Conv2b_Conv3a]
init_type=ReLU
size=3,3,1

[Conv3a]
size=48
activation=relu

[Conv3a_Conv3b]
init_type=ReLU
size=3,3,1

[Conv3b]
size=48
activation=relu
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[Conv3b_Conv4a]
init_type=ReLU
size=3,3,1

[Conv4a]
size=60
activation=relu

[Conv4a_Conv4b]
init_type=ReLU
size=3,3,1

[Conv4b]
size=60
activation=relu
filter=max
filter_size=2,2,1
filter_stride=2,2,1

[Conv4b_Conv5]
init_type=ReLU
size=3,3,1

[Conv5]
size=200
activation=relu
dropout=1
p=0.5

[Conv5_OUTPUT]
init_type=ReLU
size=1,1,1

[OUTPUT]
size=2
activation=linear
