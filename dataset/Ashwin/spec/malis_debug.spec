[INPUT1]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/7nmTraining/out2.0

[INPUT2]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/7nmTraining/out2.1

[INPUT3]
path=./experiments/Ashwin/3D_affinity/train23_test1/VeryDeep2H_w65x9/unbalanced/eta02_out100/iter_30K/output/7nmTraining/out2.2

[LABEL1]
path=./dataset/Ashwin/data/batch2
ext=label
size=512,512,170
offset=1,1,1
pptype=affinity

[MASK1]
size=512,512,170
pptype=one
ppargs=3
