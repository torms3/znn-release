# test the malis of boundary map
import emirt
import numpy as np
import time
import utils
import matplotlib.pylab as plt
#%% parameters
z = 8
# epsilone: a small number for log to avoind -infinity
eps = 0.0000001

# largest disk radius
Dm = 500
Ds = 500

# make a fake test image
is_fake = False

# whether using constrained malis
is_constrained = False

# thicken boundary of label by morphological errosion
erosion_size = 0

# a small corner
corner_size = 0

# disk radius threshold
DrTh = 0

#%% read images
if not is_fake:
    bdm = emirt.emio.imread('../experiments/zfish/VD2D/out_sample91_output_0.tif')
    lbl = emirt.emio.imread('../dataset/zfish/Merlin_label2_24bit.tif')
    raw = emirt.emio.imread('../dataset/zfish/Merlin_raw2.tif')
    lbl = emirt.volume_util.lbl_RGB2uint32(lbl)
    lbl = lbl[z,:,:]
    bdm = bdm[z,:,:]
else:
    # fake image size
    fs = 20
    bdm = np.ones((fs,fs), dtype='float32')
    bdm[3,:] = 0.5
    bdm[3,7] = 0.8
    bdm[3,3] = 0.2
    bdm[6,:] = 0.5
    bdm[6,3] = 0.2
    bdm[6,7] = 0.8
    lbl = np.zeros((fs,fs), dtype='uint32')
    lbl[:6, :] = 1
    lbl[7:, :] = 2
assert lbl.max()>1

# only a corner for test
if corner_size > 0:
    lbl = lbl[:corner_size, :corner_size]
    bdm = bdm[:corner_size, :corner_size]

# fill label holes
print "fill boundary hole..."
utils.fill_boundary_holes( lbl )

# increase boundary width
if erosion_size>0:
    print "increase boundary width"
    erosion_structure = np.ones((erosion_size, erosion_size))
    msk = np.copy(lbl>0)
    from scipy.ndimage.morphology import binary_erosion
    msk = binary_erosion(msk, structure=erosion_structure)
    lbl[msk==False] = 0


bdm.astype("float64").tofile("boundary_map.bin")
lbl.astype('float64').tofile("label_fill_hole.bin")