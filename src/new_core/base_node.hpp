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

#ifndef ZNN_BASE_NODE_HPP_INCLUDED
#define ZNN_BASE_NODE_HPP_INCLUDED

#include "../core/types.hpp"
#include "base_edge.hpp"
#include "task_manager.hpp"

namespace zi {
namespace znn {

struct node_data
{
	// feature map
    double3d_ptr f;
    double3d_ptr dEdX;

    // FFT
    complex3d_ptr fft;
    complex3d_ptr dEdX_fft;

    node_data()
        : f()
        , dEdX()
        , fft()
        , dEdX_fft()
    {}
};

class base_node
{
protected:
	zi::mutex                   m_ ;
    zi::condition_variable      cv_;

    std::size_t                 pass_no_;
    std::size_t                 waiters_;

protected:
    const std::string           name_;
    std::size_t                 layer_no_;
    std::size_t                 neuron_no_;
	
    vec3i                       size_;
    node_data                   data_;

protected:
    std::list<base_edge_ptr>    in_edges_;
    std::list<base_edge_ptr>    out_edges_;

    std::size_t                 in_received_;
    std::size_t                 out_received_;

    bool                        sends_fft_;
    bool                        receives_fft_;


public: // mutable
    zi::mutex&   mutex()        { return m_;            }
    node_data&   data()         { return data_;         }
    std::size_t& in_received()  { return in_received_;  }
    std::size_t& out_received() { return out_received_; }

public:
    std::string name()            const { return name_;             }
    std::size_t layer_number()    const { return layer_no_;         }
    std::size_t neuron_number()   const { return neuron_no_;        }
    vec3i       size()            const { return size_;             }
    std::size_t in_received()     const { return in_received_;      }
    std::size_t out_received()    const { return out_received_;     }
    std::size_t count_in_edges()  const { return in_edges_.size();  }
    std::size_t count_out_edges() const { return out_edges_.size(); }
    bool        sends_fft()       const { return sends_fft_;        }
    bool        receives_fft()    const { return receives_fft_;     }


public:
    void set_layer_number(std::size_t l)
    {
        layer_no_ = l;
    }

    std::size_t forward_priority() const
    {
        return layer_no_ * 1000 + neuron_no_;
    }

    std::size_t backward_priority() const
    {
        return 2000000000 - (layer_no_ * 1000 + neuron_no_);
    }


public:
	std::size_t run_forward(double3d_ptr f)
	{
		mutex::guard g(m_);
        data_.f = f;
        return run_forward();
	}

	std::size_t run_backward(double3d_ptr dEdX)
	{
		mutex::guard g(m_);
        data_.dEdX = dEdX;
        return run_backward();
	}

	double3d_ptr wait(std::size_t n)
    {
        mutex::guard g(m_);
        while ( pass_no_ < n )
        {
            ++waiters_;
            cv_.wait(m_);
            --waiters_;
        }

        if ( pass_no_ == n )
        {            
            return data_.f;
        }
        else
        {
            throw std::logic_error("Pass somehow skipped!");
        }
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

            run_backward();
        }
    }


protected:
    // this function should be called under the mutex guard
	virtual std::size_t run_forward()
    {
        ZI_ASSERT((in_received_==0)&&(out_received_==0));
        ++pass_no_;

        // compute FFT(F) if needed
        if ( sends_fft_ && out_edges_.size() )
        {
            data_.fft = fftw::forward(data_.f);
        }

        FOR_EACH( it, out_edges_ )
        {
            task_manager().insert(
                    zi::bind(&base_node::forward_edge, this, *it),
                    (*it)->out()->forward_priority()
                );
        }

        if ( waiters_ )
        {
            cv_.notify_all();
        }

        return pass_no_;
    }

    // this function should be called under the mutex guard
	virtual std::size_t run_backward()
    {
        ZI_ASSERT((in_received_==0)&&(out_received_==0));
        ++pass_no_;

        // compute FFT(G) if needed
        if ( receives_fft_ && in_edges_.size() )
        {
            vec3i s = in_edges_.front()->in()->size();
            data_.dEdX = volume_utils::flip(data_.dEdX);
            data_.dEdX_fft = fftw::forward_pad(data_.dEdX, s);
        }

        FOR_EACH( it, in_edges_ )
        {
            task_manager().insert(
                    zi::bind(&base_node::backward_edge, this, *it),
                    (*it)->in()->backward_priority()
                );
        }

        if ( waiters_ )
        {
            cv_.notify_all();
        }
        
        return pass_no_;
    }

protected:
    void forward_edge(base_edge_ptr e)
    {
        e->forward();
    }

    void backward_edge(base_edge_ptr e)
    {
        e->backward();
        e->update();
    }


public:
    void add_out_edge(base_edge_ptr e)
    {
        out_edges_.push_back(e);
    }

    void add_in_edge(base_edge_ptr e)
    {
        in_edges_.push_back(e);
    }

    double fan_in() const
    {
        double ret = static_cast<double>(0);
        if ( count_in_edges() > 0 )
        {
            vec3i sz = in_edges_.front()->size();
            ret  = static_cast<double>(sz[0]*sz[1]*sz[2]);
            ret *= count_in_edges();
        }
        return ret;
    }

    double fan_out() const
    {
        double ret = static_cast<double>(0);
        if ( count_out_edges() > 0 )
        {
            vec3i sz = out_edges_.front()->size();
            ret  = static_cast<double>(sz[0]*sz[1]*sz[2]);
            ret *= count_out_edges();
        }
        return ret;
    }


public:
    base_node(const std::string& name,
              std::size_t layer_no = 0,
              std::size_t neuron_no = 0)
        : m_()
        , cv_()
        , pass_no_(0)
        , waiters_(0)
        , name_(name)
        , layer_no_(layer_no)
        , neuron_no_(neuron_no)
        , size_(vec3i::zero)
        , data_()
        , in_edges_()
        , out_edges_()
        , in_received_(0)
        , out_received_(0)
        , sends_fft_(false)
        , receives_fft_(false)
    {}

    virtual ~base_node() {};

}; // class base_node

typedef boost::shared_ptr<base_node> base_node_ptr;

}} // namespace zi::znn

#endif // ZNN_BASE_NODE_HPP_INCLUDED
