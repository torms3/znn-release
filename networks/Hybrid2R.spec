[INPUT]
size=1

[INPUT2]
size=1

[INPUT_C1]
init_type=normalized
size=4,4,1
eta=0.01
mom=0.9

[INPUT2_C1]
init_type=normalized
size=4,4,1
eta=0.01
mom=0.9

[C1]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1
bias=0
eta=0.01
mom=0.9

[C1_C2]
init_type=normalized
size=4,4,1
eta=0.01
mom=0.9

[C2]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1
bias=0
eta=0.01
mom=0.9

[C2_C3]
init_type=normalized
size=4,4,1
eta=0.01
mom=0.9

[C3]
size=48
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=2,2,1
filter_stride=2,2,1
bias=0
eta=0.01
mom=0.9

[C3_FC]
init_type=normalized
size=4,4,3
eta=0.01
mom=0.9
load=0

[FC]
size=100
activation=tanh
act_params=1.7159,0.6666
filter=max
filter_size=1,1,1
filter_stride=1,1,1
bias=0
eta=0.01
mom=0.9
load=0

[FC_OUTPUT]
init_type=normalized
size=1,1,1
eta=0.01
mom=0.9
load=0

[OUTPUT]
size=3
activation=forward_logistic
init_type=zero
bias=0
eta=0.01
mom=0.9
load=0
