#ifndef ZNN_GET_SEGMENTATION_DEBUG_HPP_INCLUDED
#define ZNN_GET_SEGMENTATION_DEBUG_HPP_INCLUDED

#include "debug.hpp"
#include "cost_fn/utils/get_segmentation.hpp"
#include "cost_fn/utils/make_affinity.hpp"

namespace zi {
namespace znn {

//
// Inputs
//
//      data_path       : path to label
//      save_path       : path to save
//      outsz           : input volume size
//
class get_segmentation_debug : virtual public debug
{
private:
    std::list<double3d_ptr>     label;


public:
    virtual void run()
    {
        load_input();

        zi::wall_timer wt;
        {
            long3d_ptr seg = get_segmentation(label);

            std::cout << wt.elapsed<double>() << " secs [get_segmentation].\n";
            wt.restart();

            // save the resulting segmentation
            save_output(seg);

            std::cout << wt.elapsed<double>() << " secs [save].\n";
            wt.restart();
        }
    }


private:
    void load_input()
    {
        std::cout << "[get_segmentation_debug] load_input()" << std::endl;
        std::cout << "volume size: " << op_->outsz << std::endl;
        vec3i s = op_->outsz;

        std::cout << "loading label    [" << op_->data_path << "]" << std::endl;
        double3d_ptr lbl = volume_pool.get_double3d(s);
        STRONG_ASSERT(volume_utils::load(lbl, op_->data_path));
        label = make_affinity(lbl, 2);
    }

    void save_output(long3d_ptr seg)
    {
        volume_utils::save(seg, op_->save_path + ".segmentation");
    }


public:
    get_segmentation_debug( options_ptr op )
        : debug(op)
    {}

    virtual ~get_segmentation_debug() {}

};  // class get_segmentation_debug

}} // namespace zi::znn

#endif // ZNN_GET_SEGMENTATION_DEBUG_HPP_INCLUDED
