[INPUT1]
path=./dataset/SNEMI3D/data/batch1
ext=CLAHE
size=1024,1024,100
pptype=standard2D

[INPUT2]
path=./experiments/SNEMI3D/seunglab/VD2D/exp1/iter_30K/output/out1.1
size=1024,1024,100
pptype=transform
ppargs=-1,1

[LABEL1]
path=./dataset/SNEMI3D/data/batch1
ext=label
size=1024,1024,100
offset=1,1,1
pptype=affinity
ppargs=0.1,0.9

[MASK1]
size=1023,1023,99
offset=1,1,1
pptype=one
ppargs=3
