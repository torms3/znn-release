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

#ifndef ZNN_SUM_EDGE_HPP_INCLUDED
#define ZNN_SUM_EDGE_HPP_INCLUDED

#include "base_node.hpp"
#include "../core/volume_utils.hpp"

namespace zi {
namespace znn {

class sum_edge : virtual public base_edge
{
public:
	virtual void forward()
	{
		mutex::guard g(m_);

		if ( in_->sends_fft() )
		{
			ZI_ASSERT(out_->receives_fft());
			push_forward(in_->data().fft);
		}
		else
		{
			ZI_ASSERT(!out_->receives_fft());
			push_forward(in_->data().f);
		}
	}

	virtual void backward()
	{
		mutex::guard g(m_);

		if ( out_->receives_fft() )
		{
			ZI_ASSERT(in_->sends_fft());
			push_backward(out_->data().dEdX_fft);
		}
		else
		{
			ZI_ASSERT(!in_->sends_fft());
			push_backward(out_->data().dEdX);
		}
	}

	virtual void update() {}


// forward pass
protected:
	void push_forward(double3d_ptr f)
	{
		zi::mutex& m 	= out_->mutex();
		node_data& data = out_->data();

		while (1)
		{
			double3d_ptr old;
			{
				mutex::guard g(m);
				if ( out_->in_received() == 0 )
				{
					data.f = f;
					++out_->in_received();
                    break;
				}
				else
				{
					if ( data.f )
					{
						data.f.swap(old);
					}
					else
					{
						data.f = f;
						++out_->in_received();
		                break;
					}
				}
			}
			volume_utils::template add_to<double3d_ptr>(old, f);
		}

		out_->forward_done();
	}

	void push_forward(complex3d_ptr fft)
	{
		zi::mutex& m 	= out_->mutex(); 
		node_data& data = out_->data();

		while (1)
		{
			complex3d_ptr old;
			{
				mutex::guard g(m);
				if ( out_->in_received() == 0 )
				{
					data.fft = fft;
					++out_->in_received();
                    break;
				}
				else
				{
					if ( data.fft )
					{
						data.fft.swap(old);
					}
					else
					{
						data.fft = fft;
						++out_->in_received();
		                break;
					}
				}
			}
			volume_utils::template add_to<complex3d_ptr>(old, fft);
		}

		out_->forward_done();
	}

// backward pass
protected:
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

	void push_backward(complex3d_ptr grad_fft)
	{
		zi::mutex& m 	= in_->mutex(); 
		node_data& data = in_->data();

		{
			mutex::guard g(m);
			
			ZI_ASSERT(in_->out_received()==0);

			data.dEdX_fft = grad_fft;
			++in_->out_received();
		}

		in_->backward_done();
	}


public:
	sum_edge(base_node_ptr in, base_node_ptr out)
		: base_edge(in, out)
	{}

	virtual ~sum_edge() {}

}; // class sum_edge

typedef boost::shared_ptr<sum_edge> sum_edge_ptr;

}} // namespace zi::znn

#endif // ZNN_SUM_EDGE_HPP_INCLUDED
