# test the malis of boundary map
import time

#%% parameters
def get_params():
    pars = dict()
    # epsilone: a small number for log to avoind -infinity
    pars['eps'] = 0.0000001

    # largest disk radius
    pars['Dm'] = 500
    pars['Ds'] = 500

    # make a fake test image
    pars['is_fake'] = True

    # whether using constrained malis
    pars['is_constrained'] = False

    # thicken boundary of label by morphological errosion
    pars['erosion_size'] = 0

    # a small corner
    pars['corner_size'] = 0

    # disk radius threshold
    pars['DrTh'] = 0
    return pars

if __name__ == "__main__":
    # get the parameters
    pars = get_params()

    import data_prepare
    bdm, lbl = data_prepare.read_image(pars)

    # recompile and use cost_fn
    #print "compile the cost function..."
    #os.system('python compile.py cost_fn')
    import cost_fn
    start = time.time()
    if pars['is_constrained']:
        print "compute the constrained malis weight..."
        w, me, se = cost_fn.constrained_malis_weight_bdm_2D(bdm, lbl)
    else:
        print "compute the normal malis weight..."
        w, me, se = cost_fn.malis_weight_bdm_2D(bdm, lbl)

    elapsed = time.time() - start
    print "elapsed time is {} sec".format(elapsed)

    import malis_show
    malis_show.plot(pars, bdm, lbl, me, se)

    print "------end-----"
