[INPUT1]
path=./dataset/Ashwin/data/batch4
ext=image
size=256,256,121
pptype=standard2D

[LABEL1]
path=./dataset/Ashwin/data/batch4
ext=label
size=256,256,121
offset=1,1,1
pptype=affinity
ppargs=0.1,0.9

[MASK1]
size=255,255,120
offset=1,1,1
pptype=one
ppargs=3
