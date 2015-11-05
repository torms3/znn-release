#ifndef ZNN_MALIS_2D_DEBUG_HPP_INCLUDED
#define ZNN_MALIS_2D_DEBUG_HPP_INCLUDED

#include "debug.hpp"
#include "cost_fn/malis_2d.hpp"
#include "cost_fn/utils/make_affinity.hpp"

namespace zi {
namespace znn {

//
// Inputs
//
//      load_path       : path to boundary prediction
//      data_path       : path to label
//      save_path       : path to save
//
//      outsz           : input volume size
//
class malis_2d_debug : virtual public debug
{
private:
    std::list<double3d_ptr>     label;
    std::list<double3d_ptr>     affin;

    double high;
    double low;


public:
    virtual void run()
    {
        load_input();

        zi::wall_timer wt;
        {
            malis_return mpair = malis_2d(label, affin, high, low);

            std::cout << wt.elapsed<double>() << " secs [malis_2d].\n";
            wt.restart();

            // save the resulting MALIS weights
            save_output(mpair.first);

            std::cout << wt.elapsed<double>() << " secs [save].\n";
            wt.restart();
        }
    }


private:
    void load_input()
    {
        std::cout << "[malis_2d_debug] load_input()" << std::endl;
        std::cout << "volume size: " << op_->outsz << std::endl;
        vec3i s = op_->outsz;

        std::cout << "loading boundary [" << op_->load_path << "]" << std::endl;
        double3d_ptr aff = volume_pool.get_double3d(s);
        STRONG_ASSERT(volume_utils::load(aff, op_->load_path));
        affin = make_affinity(aff, 2);

        std::cout << "loading label    [" << op_->data_path << "]" << std::endl;
        double3d_ptr lbl = volume_pool.get_double3d(s);
        STRONG_ASSERT(volume_utils::load(lbl, op_->data_path));
        label = make_affinity(lbl, 2);
    }

    void save_output(malis_weight const & w)
    {
        volume_utils::save_tensor(w.merger, op_->save_path + ".merger");
        volume_utils::save_tensor(w.splitter, op_->save_path + ".splitter");

#if defined( DEBUG )
        volume_utils::save(w.ws_evolution, op_->save_path + ".evolution");
        volume_utils::save(w.time_step, op_->save_path + ".timestep");
#endif
    }


public:
    malis_2d_debug( options_ptr op, double h, double l )
        : debug(op)
        , high(h)
        , low(l)
    {}

    virtual ~malis_2d_debug() {}

};  // class malis_2d_debug

}} // namespace zi::znn

#endif // ZNN_MALIS_2D_DEBUG_HPP_INCLUDED
