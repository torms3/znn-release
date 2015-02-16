//
// Copyright (C) 2015  Aleksandar Zlateski <zlateski@mit.edu>
//                     Kisuk Lee           <kisuklee@mit.edu>
// ----------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef ZNN_NONLINEAR_NODE_HPP_INCLUDED
#define ZNN_NONLINEAR_NODE_HPP_INCLUDED

#include "base_node.hpp"
#include "../error_fn/error_fns.hpp"

namespace zi {
namespace znn {

struct node_param
{
    double eta;    // learning rate
    double mom;    // momentum parameter
    double v  ;    // momentum
    double wc ;    // L1 weight decay
    double szB;    // minibatch

    node_param(double _eta = 0.01, 
               double _mom = 0,
               double _wc  = 0,
               double _szB = 1)
        : eta(_eta)
        , mom(_mom)
        , v(0)
        , wc(_wc)
        , szB(_szB)
    {}
};

class nonlinear_node : virtual public base_node
{
protected:
    double          bias_;
    double          dEdB_;

    node_param      param_;
    error_fn_ptr    error_fn_;


public:
    void set_activation_function(error_fn_ptr fn)
    {
        error_fn_ = fn;
    }

    // TODO: not safe getter
    error_fn_ptr get_activation_function()
    {
        return error_fn_;
    }


public:
    virtual void forward_done()
    {
        mutex::guard g(m_);
        if ( in_received_ == in_edges_.size() )
        {
            in_received_ = 0;

            // compute IFFT(FFT(F)) if needed
            if ( receives_fft_ && in_edges_.size() )
            {
                vec3i s = in_edges_.front()->in()->size();
                double3d_ptr x = fftw::backward(data_.fft, s);
                volume_utils::normalize(x);
                data_.f = volume_utils::crop_right(x, size_);
            }
            
            if ( error_fn_ )
            {
                error_fn_->add_apply(bias_, data_.f);
            }

            run_forward();
        }
    }

    virtual void backward_done()
    {
        mutex::guard g(m_);
        if ( out_received_ == out_edges_.size() )
        {
            out_received_ = 0;

            // compute IFFT(FFT(G)) if needed
            if ( sends_fft_ && out_edges_.size() )
            {
                data_.dEdX = volume_utils::normalize_flip(
                    fftw::backward(data_.dEdX_fft, size()));
            }

            if ( error_fn_ )
            {
                data_.dEdX = error_fn_->gradient(data_.dEdX, data_.f);
            }

            update_bias();

            run_backward();
        }
    }

private:
    void update_bias()
    {
        dEdB_ = volume_utils::sum_all(data_.dEdX);

        // average gradients over the pixels in a minibatch
        double avg_dEdB = dEdB_/param_.szB;

        // momentum & weight decay        
        param_.v  = (param_.mom * param_.v);
        param_.v -= (param_.eta * param_.wc * bias_);
        param_.v -= (param_.eta * avg_dEdB);

        // bias update
        bias_ += param_.v;
    }


public:
    nonlinear_node(const std::string& name,
                   std::size_t layer_no = 0,
                   std::size_t neuron_no = 0,
                   double bias = 0,
                   double eta = 0.01,
                   double mom = 0,
                   double wc = 0,
                   double szB = 1,
                   error_fn_ptr fn = error_fn_ptr(new logistic_error_fn))
        : base_node(name, layer_no, neuron_no)
        , bias_(bias)
        , dEdB_(0)
        , param_(eta, mom, wc, szB)
        , error_fn_(fn)
    {}

    virtual ~nonlinear_node() {};

}; // class nonlinear_node

typedef boost::shared_ptr<nonlinear_node> NONLINEAR_NODE_ptr;

}} // namespace zi::znn

#endif // ZNN_NONLINEAR_NODE_HPP_INCLUDED
