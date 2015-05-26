[INPUT1]
path=./dataset/Ashwin/data/batch3
ext=image
size=512,512,169
pptype=standard2D

[INPUT2]
path=./experiments/Ashwin/2D_boundary/train23_test1/VD2D/dropout/iter_60K/output/out3.1
size=512,512,169
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/Ashwin/data/batch3
ext=label
size=512,512,169
pptype=binary_class

[MASK1]
size=512,512,169
pptype=one
ppargs=2
