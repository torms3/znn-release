[INPUT]
[INPUT_C1]
[C1]
[C1_C2]
[C2]
[C2_C3]

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

[Conv1b_C3]
init_type=ReLU
size=2,2,1

[C3]
[C3_C4]
[C4]
[C4_C5]
[C5]
[C5_C6]
[C6]
[C6_C7]

[C7]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,2
filter_stride=1,1,1

[C7_Conv4a]
init_type=ReLU
size=3,3,2

[Conv4a]
size=60
activation=relu

[Conv4a_Conv4b]
init_type=ReLU
size=3,3,2

[Conv4b]
size=60
activation=relu
filter=max
filter_size=2,2,2
filter_stride=1,1,1

[Conv4b_Conv5a]
init_type=ReLU
size=3,3,2

[Conv5a]
size=60
activation=relu

[Conv5a_Conv5b]
init_type=ReLU
size=3,3,2

[Conv5b]
size=100
activation=relu
dropout=1
p=0.5

[Conv5b_OUTPUT]
init_type=ReLU
size=1,1,1

[OUTPUT]
size=2
activation=linear
