#!/usr/bin/env python
__doc__ = """

Jingpeng Wu <jingpeng.wu@gmail.com>, 2015
"""
import numpy as np
import emirt
# numba accelaration
#from numba import jit

def get_cls(props, lbls):
    """
    compute classification error.

    Parameters
    ----------
    props : dict of array, network propagation output volumes.
    lbls  : dict of array, ground truth

    Returns
    -------
    c : number of classification error
    """
    c = 0.0
    for name, prop in props.iteritems():
        lbl = lbls[name]
        c = c + np.count_nonzero( (prop>0.5)!= lbl )

    return c

#@jit(nopython=True)
def square_loss(props, lbls):
    """
    compute square loss

    Parameters
    ----------
    props: numpy array, forward pass output
    lbls:  numpy array, ground truth labeling

    Return
    ------
    err:   cost energy
    grdts: numpy array, gradient volumes
    """
    grdts = dict()
    err = 0
    for name, prop in props.iteritems():
        lbl = lbls[name]
        grdt = prop - lbl
        # cost and classification error
        err = err + np.sum( grdt * grdt )
        grdts[name] = grdt * 2

    return (props, err, grdts)

#@jit(nopython=True)
def binomial_cross_entropy(props, lbls):
    """
    compute binomial cost

    Parameters
    ----------
    props:  dict of network output arrays
    lbls:   dict of ground truth arrays

    Return
    ------
    err:    cost energy
    grdts:  dict of gradient volumes
    """
    grdts = dict()
    err = 0
    for name, prop in props.iteritems():
        lbl = lbls[name]
        grdts[name] = prop - lbl
        err = err + np.sum(  -lbl*np.log(prop) - (1-lbl)*np.log(1-prop) )
    return (props, err, grdts)

#@jit(nopython=True)
def softmax(props):
    """
    softmax activation

    Parameters:
    props:  numpy array, net forward output volumes

    Returns:
    ret:   numpy array, softmax activation volumes
    """
    ret = dict()
    for name, prop in props.iteritems():
        # make sure that it is the output of binary class
        assert(prop.shape[0]==2)

        # rebase the prop for numerical stability
        # mathematically, this does not affect the softmax result!
        propmax = np.max(prop, axis=0)
        for c in xrange( prop.shape[0] ):
            prop[c,:,:,:] -= propmax

        prop = np.exp(prop)
        pesum = np.sum(prop, axis=0)
        ret[name] = np.empty(prop.shape, dtype=prop.dtype)
        for c in xrange(prop.shape[0]):
            ret[name][c,:,:,:] = prop[c,:,:,:] / pesum
    return ret

def multinomial_cross_entropy(props, lbls):
    """
    compute multinomial cross entropy

    Parameters
    ----------
    props:    list of forward pass output
    lbls:     list of ground truth labeling

    Return
    ------
    err:    cost energy
    cls:    classfication error
    grdts:  list of gradient volumes
    """
    grdts = dict()
    err = 0
    for name, prop in props.iteritems():
        lbl = lbls[name]
        grdts[name] = prop - lbl
        err = err + np.sum( -lbl * np.log(prop) )
    return (props, err, grdts)

def softmax_loss(props, lbls):
    props = softmax(props)
    return multinomial_cross_entropy(props, lbls)

def softmax_loss2(props, lbls):
    grdts = dict()
    err = 0

    for name, prop in props.iteritems():
        # make sure that it is the output of binary class
        assert(prop.shape[0]==2)

        print "original prop: ", prop

        # rebase the prop for numerical stabiligy
        # mathimatically, this do not affect the softmax result!
        # http://ufldl.stanford.edu/tutorial/supervised/SoftmaxRegression/
#        prop = prop - np.max(prop)
        propmax = np.max(prop, axis=0)
        prop[0,:,:,:] -= propmax
        prop[1,:,:,:] -= propmax

        log_softmax = np.empty(prop.shape, dtype=prop.dtype)
        log_softmax[0,:,:,:] = prop[0,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
        log_softmax[1,:,:,:] = prop[1,:,:,:] - np.logaddexp( prop[0,:,:,:], prop[1,:,:,:] )
        prop = np.exp(log_softmax)
        props[name] = prop

        lbl = lbls[name]
        grdts[name] = prop - lbl
        err = err + np.sum( -lbl * log_softmax )
        print "gradient: ", grdts[name]
        assert(not np.any(np.isnan(grdts[name])))
    return (props, err, grdts)

#def hinge_loss(props, lbls):
# TO-DO


def malis_find(sid, seg):
    """
    find the root/segment id
    """
    return seg[sid-1]

def malis_union(r1, r2, seg, id_sets):
    """
    union the two sets, keep the "tree" update.

    parameters
    ----------
    r1, r2 : segment id of two sets
    seg : 1d array, record the segment id of all the segments
    id_sets : dict, key is the root/segment id
                    value is a set of voxel ids.

    returns
    -------
    seg :
    id_sets
    """
    # get set pair
    set1 = id_sets[r1]
    set2 = id_sets[r2]

    # make sure that set1 is larger than set2
    if len(set1) < len(set2):
        r1, r2 = r2, r1
        set1, set2 = set2, set1
    # merge the small set to big set
    id_sets[r1]= set1.union( set2 )
    # update the segmentation
    for vid in set2:
        seg[vid-1] = r1
    # remove the small set in dict
    del id_sets[r2]
    return seg, id_sets

# TO-DO, not fully implemented
def malis_weight_aff(affs, true_affs, threshold=0.5):
    """
    compute malis tree_size

    Parameters:
    -----------
    affs:      4D array of forward pass output affinity graphs, size: C*Z*Y*X
    true_affs : 4d array of ground truth affinity graph
    threshold: threshold for segmentation

    Return:
    ------
    weights : 4D array of weights
    """
    # segment the true affinity graph
    tseg = emirt.volume_util.aff2seg(true_affs)
    tid_sets, tsegf = emirt.volume_util.map_set( tseg )

    if isinstance(affs, dict):
        assert( len(affs.keys())==1 )
        key = affs.keys()[0]
        affs = affs.values()[0]
    # get affinity graphs
    xaff = affs[2]
    yaff = affs[1]
    zaff = affs[0]
    shape = xaff.shape

    # initialize segmentation with individual label of each voxel
    seg_shp = np.asarray(xaff.shape)+1
    N = np.prod( seg_shp )
    ids = np.arange(1, N+1).reshape( seg_shp )
    seg = np.copy( ids ).flatten()

    # initialize edges: aff, id1, id2, z/y/x, true_aff
    edges = list()
    for z in xrange(shape[0]):
        for y in xrange(shape[1]):
            for x in xrange(1,shape[2]):
                edges.append( (xaff[z,y,x], ids[z,y,x], ids[z,y,x-1], 2) )
    for z in xrange(shape[0]):
        for y in xrange(1,shape[1]):
            for x in xrange(shape[2]):
                edges.append( (yaff[z,y,x], ids[z,y,x], ids[z,y-1,x], 1) )
    for z in xrange(1,shape[0]):
        for y in xrange(shape[1]):
            for x in xrange(shape[2]):
                edges.append( (zaff[z,y,x], ids[z,y,x], ids[z-1,y,x], 0) )
    # descending sort
    edges.sort(reverse=True)

    # find the maximum-spanning tree based on union-find algorithm
    weights = np.zeros( affs.shape, dtype=affs.dtype )
    # the dict set containing all the sets
    id_sets = dict()
    # initialize the id sets
    for i in xrange(1,N+1):
        id_sets[i] = i

    # accumulate the merge and split errors
    #merr = serr = 0

    # union find the sets
    for e in edges:
        # find the segment/root id
        r1 = malis_find(e[1], seg)
        r2 = malis_find(e[2], seg)

        if r1==r2:
            # this is not a maximin edge
            continue

        if e[0]>threshold:
            # merge two sets, the voxel pairs not in a segments are errors
            #weights[e[3], r1-1] = weights[e[3],r1-1] + s1*s2
            # merge the two sets/trees
            pass

        else:
        # do not merge, the voxel pairs in a same segments are errors
            print "dp "


    # normalize the weights
    #N = float(N)
    #weights = weights * (3*N) / ( N*(N-1)/2 )
    #weights = weights.reshape( affs.shape )
    # transform to dictionary
    ret = dict()
    #ret[key] = weights
    return ret

class CDomain:
    def __init__( lid=None, vid=None ):
        """
        lid: label id
        vid: voxel id
        """
        import numbers
        # a dictionary containing voxel number of different segment id
        sizes = dict()
        # a dictionary containing voxel sets
        sets = set()
        total_size = 0
        if isinstance(id0, numbers.Number):
            sizes[lid] = 1
            sets[lid]  = {vid}

    def union(self, dm2 ):
        """
        merge with another domain
        dm2: CDomain, another domain
        """
        for id2, sz2 in dm2.sizes.iteritems():
            set2 = dm2.sets[id2]
            if self.sizes.has_key(id2):
                # have common segment id, merge together
                self.sizes[id2] += sz2
                self.sets[id2] = self.sets[id2].union( set2 )
            else:
                # do not have common id, create new one
                self.sizes[id2] = sz2
                self.sets[id2] = set2
        return
    def clear(self):
        """
        delete all the containt
        """
        self.sizes = dict()
        self.sets = dict()
        return

    def find( vid ):
        """
        find whether this voxel is in this domain
        """
        for s in self.sets.values():
            if vid in s:
                return True
        return False

    def get_merge_split_errors(self, dm2):
        """
        compute the merge and split error of two domains
        """
        # merging and splitting error
        me = 0
        se = 0
        for lid1, sz1 in self.sizes.iteritems():
            for lid2, sz2 in dm2.sizes.iteritems():
                if lid1==lid2:
                    # they should be merged together
                    # this is a split error
                    se += sz1 * sz2
                else:
                    # they should be splitted
                    # this is a merging error
                    me += sz1 * sz2
        return me, se

class CDomains:
    """
    the list of watershed domains.
    """
    def __init__( self, lbl ):
        """
        Parameters
        ----------
        lbl: 2D/3D array, manual label image
        """
        assert(lbl.ndim==2 or lbl.ndim==3)
        dms = list()
        # voxel id start from 0
        for vid in xrange( lbl.size ):
            lid = lbl.flat[vid]
            dms.append( CDomain(lid, vid) )
        return

    def find( self, vid ):
        """
        find the corresponding domain of a voxel
        vid: voxel ID
        Return
        ------
        dm: corresponding watershed domain
        """
        for i, dm in enumerate( self.dms ):
            if dm.find(vid):
                return i, dm
        raise NameError("the voxel id was not found!")
        return

    def union(vid1, vid2):
        """
        union the two watershed domain of two voxel ids
        """
        i1, dm1 = self.find(vid1)
        i2, dm2 = self.find(vid2)
        if i1 != i2:
            # they are in different domains
            me, se = dm1.get_merge_split_errors( dm2 )
            # merge these two domains
            self.dms[i1] = dm1.union( dm2 )
            self.dms.pop(i2)
            return me, se
        else:
            return 0,0

def malis_weight_bdm_2D(bdm, lbl, threshold=0.5):
    """
    compute malis weight for boundary map

    Parameters
    ----------
    bdm: 2D array, forward pass output boundary map
    lbl: 2D array, manual labels containing segment ids
    threshold: binarization threshold

    Returns
    -------
    weights: 2D array of weights
    """
    # eliminate the second output
    assert(bdm.ndim==2)
    assert(bdm.shape==lbl.shape)

    # initialize segmentation with individual id of each voxel
    # voxel id start from 0, is exactly the coordinate of voxel in 1D
    vids = np.arange(bdm.size).reshape( bdm.shape )

    # create edges: bdm, id1, id2, true label
    # the affinity of neiboring boundary map voxels
    # was represented by the minimal boundary map value

    edges = list()
    for y in xrange(bdm.shape[0]):
        for x in xrange(bdm.shape[1]-1):
            bmv1 = bdm[y,x]
            vid1 = vids[y,x]
            bmv2 = bdm[y,x+1]
            vid2 = vids[y,x+1]
            # the voxel with id1 will always has the minimal value
            if bmv1 > bmv2:
                bmv1, bmv2 = bmv2, bmv1
                vid1, vid2 = vid2, vid1
            edges.append((bmv1, vid1, vid2))

    for y in xrange(bdm.shape[1]-1):
        for x in xrange(bdm.shape[0]):
            # boundary map value and voxel id
            bmv1 = bdm[y,x]
            vid1 = vids[y,x]
            bmv2 = bdm[y+1,x]
            vid2 = vids[y+1,x]
            if bmv1 > bmv2:
                bmv1, bmv2 = bmv2, bmv1
                vid1, vid2 = vid2, vid1
            edges.append((bmv1, vid1, vid2))

    # descending sort
    edges.sort(reverse=True)

    # initalize the merge and split errors
    merr = np.zeros(bdm.size, dtype=bdm.dtype)
    serr = np.zeros(bdm.size, dtype=bdm.dtype)

    # initalize the watershed domains
    dms = CDomains( lbl )

    # find the maximum spanning tree based on union-find algorithm
    for e in edges:
        # voxel ids
        vid1 = e[1]
        vid2 = e[2]
        # union the domains
        me, se = dms.union( vid1, vid2 )

        # deal with the maximin edge
        # accumulate the merging error
        merr[vid1] += me
        # accumulate the spliting error
        serr[vid2] += se

    # normalize the weight
    merr *= merr.size*0.5 / np.sum(merr, dtype=bdm.dtype)
    serr *= serr.size*0.5 / np.sum(serr, dtype=bdm.dtype)
    # reshape the err

    merr = merr.reshape(bdm.shape)
    serr = serr.reshape(bdm.shape)
    # combine the two error weights
    w = (merr + serr)

    return (w, merr, serr)

def constrain_label(bdm, lbl):
    # merging error boundary map filled with intracellular ground truth
    mbdm = np.copy(bdm)
    mbdm[lbl>0] = 1

    # splitting error boundary map filled with boundary ground truth
    sbdm = np.copy(bdm)
    sbdm[lbl==0] = 0

    return mbdm, sbdm

def constrained_malis_weight_bdm_2D(bdm, lbl, threshold=0.5):
    """
    adding constraints for malis weight
    fill the intracellular space with ground truth when computing merging error
    fill the boundary with ground truth when computing spliting error
    """
    mbdm, sbdm = constrain_label(bdm, lbl)
    # get the merger weights
    mw, mme, mse = malis_weight_bdm_2D(mbdm, lbl, threshold)
    # get the splitter weights
    sw, sme, sse = malis_weight_bdm_2D(sbdm, lbl, threshold)
    w = mme + sse
    return (w, mme, sse)

def malis_weight_bdm(bdm, lbl, threshold=0.5):
    """
    compute the malis weight of boundary map

    Parameter
    ---------
    bdm: 3D or 4D array, boundary map
    lbl: 3D or 4D array, binary ground truth

    Return
    ------
    weights: 3D or 4D array, the malis weights
    """
    assert(bdm.shape==lbl.shape)
    assert(bdm.ndim==4 or bdm.ndim==3)
    original_shape = bdm.shape
    if bdm.ndim==3:
        bdm = bdm.reshape((1,)+(bdm.shape))
        lbl = lbl.reshape((1,)+(lbl.shape))

    # only compute weight of the first channel
    bdm0 = bdm[0,:,:,:]
    # segment the ground truth label
    lbl0 = emirt.volume_util.bdm2seg(lbl[0,:,:,:])

    # initialize the weights
    weights = np.empty(bdm.shape, dtype=bdm.dtype)
    # traverse along the z axis
    for z in xrange(bdm.shape[1]):
        w = malis_weight_bdm_2D(bdm0[z,:,:], lbl0[z,:,:], threshold)
        for c in xrange(bdm.shape[0]):
            weights[c,z,:,:] = w
    weights = weights.reshape( original_shape )

    return weights

def malis_weight(props, lbls):
    """
    compute the malis weight including boundary map and affinity cases
    """
    ret = dict()
    for name, prop in props.iteritems():
        lbl = lbls[name]
        if prop.shape[0]==3:
            # affinity output
            ret[name] = malis_weight_aff(prop, lbl)
        else:
            # take it as boundary map
            ret[name] = malis_weight_bdm(prop, lbl)
    return ret

def sparse_cost(outputs, labels, cost_fn):
    """
    Sparse Versions of Pixel-Wise Cost Functions

    Parameters
    ----------
    outputs: numpy array, forward pass output
    labels:  numpy array, ground truth labeling
    cost_fn: function to make sparse

    Return
    ------
    err:   cost energy
    grdts: numpy array, gradient volumes
    """

    flat_outputs = outputs[labels != 0]
    flat_labels = labels[labels != 0]

    errors, gradients = cost_fn(flat_outputs, flat_labels)

    # full_errors = np.zeros(labels.shape)
    full_gradients = np.zeros(labels.shape)

    # full_errors[labels != 0] = errors
    full_gradients[labels != 0] = gradients

    return (errors, full_gradients)
