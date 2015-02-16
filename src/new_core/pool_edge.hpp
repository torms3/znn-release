//
// Copyright (C) 2015  Aleksandar Zlateski <zlateski@mit.edu>
//					   Kisuk Lee           <kisuklee@mit.edu>
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

#ifndef ZNN_POOL_EDGE_HPP_INCLUDED
#define ZNN_POOL_EDGE_HPP_INCLUDED

#include "base_node.hpp"
#include "../core/volume_utils.hpp"
#include "../core/generic_filter.hpp"

namespace zi {
namespace znn {

class pool_edge : virtual public base_edge
{
protected:
	// zi::function<bool(double,double)> filter_function_;	

protected:
	long3d_ptr   	indices_;
    vec3i 			stride_ ;	// pooling filter stride
    vec3i 			sparse_	;	// pooling filter sparseness


public:
	virtual vec3i real_size() const
    {
    	return (size_ - vec3i::one)*sparse_ + vec3i::one;
    }

public:
    vec3i stride() const { return stride_; }
    vec3i sparse() const { return sparse_; }
    

public:
	virtual void forward()
	{
		mutex::guard g(m_);

		node_data& data = in_->data();

		double3d_ptr filtered = data.f;

		// filtering
		// TODO: currently hard-coded to use max-filtering
        if ( size_ != vec3i::one )
        {
            std::pair<double3d_ptr, long3d_ptr> fandi =
                generic_filter(data.f, size_, sparse_, std::greater<double>());
            
            filtered = fandi.first;
            indices_ = fandi.second;
        }

        push_forward(filtered);
	}

	virtual void backward()
	{
		mutex::guard g(m_);

		node_data& data = out_->data();

		double3d_ptr switched = data.dEdX;

		// filter backprop
		if ( size_ != vec3i::one )
		{
			STRONG_ASSERT(indices_);
			switched = do_filter_backprop(data.dEdX, indices_, real_size());
		}

		push_backward(switched);
	}

	virtual void update() {}


protected:
	void push_forward(double3d_ptr f)
	{
		zi::mutex& m 	= out_->mutex();
		node_data& data = out_->data();
		
		{
			mutex::guard g(m);

			ZI_ASSERT(out_->in_received()==0);
			
			data.f = f;
			++out_->in_received();
		}

		out_->forward_done();
	}

	void push_backward(double3d_ptr grad)
	{
		zi::mutex& m 	= in_->mutex();
		node_data& data = in_->data();

		{
			mutex::guard g(m);

			ZI_ASSERT(in_->out_received()==0);
			
			data.dEdX = grad;
			++in_->out_received();
		}

		in_->backward_done();
	}


public:
	pool_edge(base_node_ptr in, base_node_ptr out,
			  vec3i stride = vec3i::one,
			  vec3i sparse = vec3i::one)
		: base_edge(in, out)
		, indices_()
		, stride_(stride)
		, sparse_(sparse)
	{}

	virtual ~pool_edge() {}

}; // class pool_edge

typedef boost::shared_ptr<pool_edge> pool_edge_ptr;

}} // namespace zi::znn

#endif // ZNN_POOL_EDGE_HPP_INCLUDED
