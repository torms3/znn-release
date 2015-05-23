[INPUT1]
path=./dataset/SNEMI3D/data/batch2
ext=CLAHE
size=1024,1024,100
pptype=standard2D

[LABEL1]
path=./dataset/SNEMI3D/data/batch2
ext=label
size=1024,1024,100
pptype=binarify

[MASK1]
size=1024,1024,100
pptype=one
ppargs=2
