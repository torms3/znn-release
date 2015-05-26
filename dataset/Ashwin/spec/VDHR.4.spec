[INPUT1]
path=./dataset/Ashwin/data/batch4
ext=image
size=256,256,121
pptype=standard2D

[INPUT2]
path=./experiments/Ashwin/2D_boundary/train23_test1/VD2D/dropout/iter_60K/output/out4.1
size=256,256,121
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/Ashwin/data/batch4
ext=label
size=256,256,121
pptype=binary_class

[MASK1]
size=256,256,121
pptype=one
ppargs=2
