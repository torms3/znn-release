#ifndef ZNN_MALIS_DEBUG_HPP_INCLUDED
#define ZNN_MALIS_DEBUG_HPP_INCLUDED

#include "debug.hpp"
#include "cost_fn/malis.hpp"

namespace zi {
namespace znn {

//
// Inputs
//
//      data_path       : path to data spec
//      save_path       : path to save
//      outname         : filename to save
//      n_iters         : number of iteration
//      outsz           : output patch size
//      subvol_dim      : receptive field (dummy)
//      cost_fn_param   : margin for square-square loss
//
class malis_debug : virtual public debug
{
public:
    virtual void run()
    {
        double margin = op_->cost_fn_param;
        std::cout << "margin = " << margin << std::endl;

        load_input(op_->data_path);

        malis_pair mpair;

        zi::wall_timer wt;
        for ( std::size_t i = 0; i < op_->n_iters ; ++i )
        {
            std::cout << "[Iter: " << std::setw(count_digit(op_->n_iters))
                                   << i + 1 << "] ";

            // random sampling
            sample_ptr s = dp_->random_sample();

            // applying MALIS
            for ( std::size_t j = 0; j < 5; ++j )
            {
                margin = static_cast<double>(j)*0.1;
                mpair = malis(s->labels, s->inputs, s->masks, margin);

                // accumulate volume
                const std::string& hstr = op_->save_path + op_->outname;
                const std::string& mstr = ".m" + boost::lexical_cast<std::string>(j);

                accumulate_volumes(s->inputs,   hstr + mstr + ".affin", i+1);
                accumulate_volumes(s->labels,   hstr + mstr + ".truth", i+1);
                accumulate_volumes(mpair.first, hstr + mstr + ".malis", i+1);
            }

            // report elapsed time
            std::cout << wt.elapsed<double>() << " secs/update\n";
            wt.restart();
        }
    }


private:
    void load_input( const std::string& fname )
    {
        std::cout << "[malis_debug] load_input()" << std::endl;

        std::vector<vec3i> in_szs;
        std::vector<vec3i> out_szs;

        vec3i in_sz  = op_->outsz + op_->subvol_dim - vec3i::one;
        vec3i out_sz = op_->outsz;

        std::cout << "IN:  " << in_sz << std::endl;
        std::cout << "OUT: " << out_sz << std::endl;

        for ( std::size_t i = 0; i < 3; ++i )
        {
            in_szs.push_back(in_sz);
            out_szs.push_back(out_sz);
        }

        std::cout << "loading [" << fname << "]" << std::endl;

        affinity_data_provider* dp =
            new affinity_data_provider(fname,in_szs,out_szs);

        dp->data_augmentation(false);

        dp_ = data_provider_ptr(dp);
    }

    void accumulate_volumes( std::list<double3d_ptr> v,
                             const std::string& fname,
                             std::size_t counter )
    {
        int i = 0;
        FOR_EACH( it, v )
        {
            std::ostringstream subname;
            subname << fname << "." << i++;
            volume_utils::save_append((*it), subname.str());
        }

        // volume size info
        export_size_info(size_of(v.front()), counter, fname);
    }


public:
    malis_debug( options_ptr op )
        : debug(op)
    {}

    virtual ~malis_debug(){};

};  // class malis_debug

}} // namespace zi::znn

#endif // ZNN_MALIS_DEBUG_HPP_INCLUDED
