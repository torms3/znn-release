//
// Copyright (C) 2015  Kisuk Lee           <kisuklee@mit.edu>
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

#ifndef ZNN_MAXOUT_EDGE_HPP_INCLUDED
#define ZNN_MAXOUT_EDGE_HPP_INCLUDED

#include "maxout_node.hpp"
#include "../core/volume_utils.hpp"

namespace zi {
namespace znn {

class maxout_edge : virtual public base_edge
{
protected:
	std::size_t	group_;


public:
	virtual void forward()
	{
		mutex::guard g(m_);

		double3d_ptr f = in_->data().f;
		long3d_ptr   p = volume_pool.get_long3d(f);
        volume_utils::fill_n(p, static_cast<int64_t>(group_));

        push_forward(f, p);
	}

	virtual void backward()
	{
		mutex::guard g(m_);

		maxout_node_ptr out = 
			boost::dynamic_pointer_cast<maxout_node>(out_);
		STRONG_ASSERT(out);

		int64_t idx 		= static_cast<int64_t>(group_);
		double3d_ptr grad 	= out->data().dEdX;
		long3d_ptr& maxidx  = out->maxout_indices();

        push_backward(volume_utils::max_backprop(grad, maxidx, idx));
	}

	virtual void update() {}


protected:
	void push_forward(double3d_ptr f, long3d_ptr p)
	{
		maxout_node_ptr out = 
			boost::dynamic_pointer_cast<maxout_node>(out_);
		STRONG_ASSERT(out);

		zi::mutex&  m 		= out->mutex();
		node_data&  data 	= out->data();
		long3d_ptr& maxidx	= out->maxout_indices();

		while (1)
		{
			double3d_ptr oldf;
			long3d_ptr   oldp;
			{
				mutex::guard g(m);
				if ( out->in_received() == 0 )
				{
					data.f = f;
					maxidx = p;
					++out->in_received();
                    break;
				}
				else
				{
					if ( data.f )
					{
						data.f.swap(oldf);
						maxidx.swap(oldp);
					}
					else
					{
						data.f = f;
						maxidx = p;
						++out->in_received();
		                break;
					}
				}
			}
			volume_utils::maxout(oldf, oldp, f, p);
		}

		out->forward_done();
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
	maxout_edge(base_node_ptr in, base_node_ptr out,
				std::size_t group)
		: base_edge(in, out)
		, group_(group)
	{}

	virtual ~maxout_edge() {}

}; // class maxout_edge

typedef boost::shared_ptr<maxout_edge> maxout_edge_ptr;

}} // namespace zi::znn

#endif // ZNN_MAXOUT_EDGE_HPP_INCLUDED
