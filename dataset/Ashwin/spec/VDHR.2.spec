[INPUT1]
path=./dataset/Ashwin/data/batch2
ext=image
size=512,512,170
pptype=standard2D

[INPUT2]
path=./experiments/Ashwin/2D_boundary/train23_test1/VD2D/dropout/iter_60K/output/out2.1
size=512,512,170
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/Ashwin/data/batch2
ext=label
size=512,512,170
pptype=binary_class

[MASK1]
size=512,512,170
pptype=one
ppargs=2
