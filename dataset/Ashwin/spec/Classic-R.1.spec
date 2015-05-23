[INPUT1]
path=./dataset/Ashwin/data/batch1
ext=image
size=255,255,168
pptype=standard2D

[INPUT2]
path=./experiments/Ashwin/2D_boundary/train23_test1/Base2D/unbalanced/iter_30K/output/out1.1
size=255,255,168
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/Ashwin/data/batch1
ext=label
size=255,255,168
pptype=binary_class

[MASK1]
size=255,255,168
pptype=one
ppargs=2
