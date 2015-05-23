[INPUT1]
path=./dataset/SNEMI3D/data/batch3
ext=CLAHE
size=1024,1024,70
pptype=standard2D

[LABEL1]
path=./dataset/SNEMI3D/data/batch3
ext=label
size=1024,1024,70
pptype=binarify

[MASK1]
size=1024,1024,70
pptype=one
ppargs=2
