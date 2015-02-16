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

#ifndef ZNN_CONV_EDGE_HPP_INCLUDED
#define ZNN_CONV_EDGE_HPP_INCLUDED

#include "base_node.hpp"
#include "../core/volume_utils.hpp"
#include "../core/bf_conv.hpp"

namespace zi {
namespace znn {

class conv_edge : virtual public base_edge
{
protected: // data
	double3d_ptr    W_     ;	// convolution filter
	double3d_ptr	dEdW_  ;	// gradient(W)
	double3d_ptr    V_     ;    // momentum(W)
    complex3d_ptr   fft_   ;	// FFT(W)
    vec3i			stride_;	// convolution filter stride
    vec3i           sparse_;	// convolution filter sparseness
    
protected: // parameter
    double          norm_  ;	// weight L2-norm
    double          eta_   ;    // learning rate parameter
    double          mom_   ;    // momentum parameter
    double          wc_    ;    // L1 weight decay
    double          szB_   ;    // minibatch size


public:
	virtual vec3i real_size() const
    {
    	return (size_ - vec3i::one)*sparse_ + vec3i::one;
    }

public:
    vec3i stride() const { return stride_; }
    vec3i sparse() const { return sparse_; }

public:
	// initialization, load, ...
	void reset_W(double3d_ptr W)
    {
        if ( size_of(W) == size_ )
        {
            W_ = volume_utils::sparse_decompress(W, sparse_);
        }
        else
        {
            std::cout << "reset_W: size mismatch." << std::endl;
        }
    }

    void reset_V()
    {
        volume_utils::zero_out(V_);
    }

    void set_sparse(vec3i s)
    {
        if ( sparse_ != s )
        {
        	if ( sparse_ != vec3i::one )
        	{
        		// TODO: sparse_compress() is not verified yet.
        		std::cout << "sparse_compress!" << std::endl;
	        	W_ = volume_utils::sparse_compress(W_, sparse_);
	            V_ = volume_utils::sparse_compress(V_, sparse_);
	        }

            sparse_ = s;

            if ( s != vec3i::one )
            {
                W_ = volume_utils::sparse_decompress(W_, s);
                V_ = volume_utils::sparse_decompress(V_, s);
            }
        }
    }

public:
	double norm() const  { return norm_; }
	double eta()  const  { return eta_ ; }
	double mom()  const  { return mom_ ; }
	double wc()   const  { return wc_  ; }
	double szB()  const  { return szB_ ; }

	void eta(double eta) { eta_ = std::max(0.0, std::min(1.0, eta)); }
	void mom(double mom) { mom_ = std::max(0.0, std::min(1.0, mom)); }
	void wc(double wc)   { wc_  = std::max(0.0, std::min(1.0, wc)) ; }
	void szB(double szB) { szB_ = std::max(1.0, szB)			   ; }


public:
	virtual void forward()
	{
		if ( in_->sends_fft() )
		{
			ZI_ASSERT(out_->receives_fft());
			forward_fft(in_->data().fft);
		}
		else
		{
			ZI_ASSERT(!out_->receives_fft());
			forward_direct(in_->data().f);
		}
	}

	virtual void backward()
	{
		if ( out_->receives_fft() )
		{
			ZI_ASSERT(in_->sends_fft());
			backward_fft(out_->data().dEdX_fft);
		}
		else
		{
			ZI_ASSERT(!in_->sends_fft());
			backward_direct(out_->data().dEdX);
		}
	}

	virtual void update()
	{
		mutex::guard g(m_);

		STRONG_ASSERT(dEdW_);

		// update weight
        volume_utils::elementwise_mul_by(V_, mom_);
        volume_utils::elementwise_div_by(dEdW_, szB_);
        volume_utils::mul_add_to(-(eta_), dEdW_, V_);
        volume_utils::mul_add_to(-(wc_*eta_), W_, V_);
        volume_utils::add_to(V_, W_);

        // sparse filter
        if ( sparse_ != vec3i::one )
        {
            W_ = volume_utils::zero_out_nongrid(W_, sparse_);
        }
        
        // updated L2 norm
        norm_ = volume_utils::square_sum(W_);

        // FFT(W)
        fft_.reset();
        if ( in_->sends_fft() && out_->receives_fft() )
        {
            fft_ = fftw::forward_pad(W_, in_->size());
        }
	}


// forward pass
private:
	void forward_direct(double3d_ptr f)
	{
		mutex::guard g(m_);

		double3d_ptr convolved;

		if ( size_ == vec3i::one )
		{
			convolved = bf_conv_constant(f, (*W_)[0][0][0]);
		}
		else
		{
			if ( sparse_ == vec3i::one )
			{
				convolved = bf_conv(f, W_);
			}
			else
			{
				convolved = bf_conv_sparse(f, W_, sparse_);
			}
		}

		push_forward(convolved);
	}

	void forward_fft(complex3d_ptr f)
	{
		mutex::guard g(m_);

		// compute FFT(W) if needed
		if ( !fft_ )
		{
			fft_ = fftw::forward_pad(W_, size_of(f));
		}

		push_forward(volume_utils::elementwise_mul(f, fft_));
	}

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
private:
	void backward_direct(double3d_ptr dEdX)
	{
		mutex::guard g(m_);

		double3d_ptr grad;

		// TODO: no need to compute grad when it's toward input node

		if ( size_ == vec3i::one )
		{
			dEdW_ = volume_pool.get_double3d(vec3i::one);
			(*dEdW_)[0][0][0] = bf_conv_flipped_constant(in_->data().f, dEdX);
			grad = bf_conv_inverse_constant(dEdX, (*W_)[0][0][0]);
		}
		else
		{
			if ( sparse_ == vec3i::one )
			{
				dEdW_ = bf_conv_flipped(in_->data().f, dEdX);
                grad  = bf_conv_inverse(dEdX, W_);
			}
			else
			{
				dEdW_ = bf_conv_flipped_sparse(in_->data().f, dEdX, sparse_);
                grad  = bf_conv_inverse_sparse(dEdX, W_, sparse_);
			}
		}

		push_backward(grad);
	}

	void backward_fft(complex3d_ptr dEdX_fft)
	{
		mutex::guard g(m_);

		complex3d_ptr dEdW_fft
            = volume_utils::elementwise_mul(in_->data().fft, dEdX_fft);

        vec3i s = in_->size();
        dEdW_ = fftw::backward(dEdW_fft, s);
        dEdW_ = volume_utils::normalize_flip(dEdW_);
        dEdW_ = volume_utils::crop_left(dEdW_, real_size());

		push_backward(volume_utils::elementwise_mul(dEdX_fft, fft_));
	}

	void push_backward(double3d_ptr grad)
	{
		zi::mutex& m 	= in_->mutex();
		node_data& data = in_->data();

		while (1)
		{
			double3d_ptr old;
			{
				mutex::guard g(m);
				if ( in_->out_received() == 0 )
				{
					data.dEdX = grad;
					++in_->out_received();
                    break;
				}
				else
				{
					if ( data.dEdX )
					{
						data.dEdX.swap(old);
					}
					else
					{
						data.dEdX = grad;
						++in_->out_received();
		                break;
					}
				}
			}
			volume_utils::template add_to<double3d_ptr>(old, grad);
		}

		in_->backward_done();
	}

	void push_backward(complex3d_ptr grad_fft)
	{
		zi::mutex& m 	= in_->mutex(); 
		node_data& data = in_->data();

		while (1)
		{
			complex3d_ptr old;
			{
				mutex::guard g(m);
				if ( in_->out_received() == 0 )
				{
					data.dEdX_fft = grad_fft;
					++in_->out_received();
                    break;
				}
				else
				{
					if ( data.dEdX_fft )
					{
						data.dEdX_fft.swap(old);
					}
					else
					{
						data.dEdX_fft = grad_fft;
						++in_->out_received();
		                break;
					}
				}
			}
			volume_utils::template add_to<complex3d_ptr>(old, grad_fft);
		}

		in_->backward_done();
	}


// for exporting net
public:
    void print_W(std::ostream& stream)
    {
        for ( std::size_t z = 0; z < W_->shape()[2]; z += sparse_[2] )
            for ( std::size_t y = 0; y < W_->shape()[1]; y += sparse_[1] )
                for ( std::size_t x = 0; x < W_->shape()[0]; x += sparse_[0] )
                {
                    double d = (*W_)[x][y][z];
                    stream.write(reinterpret_cast<char*>(&d), sizeof(double));
                }
    }


private:
	void init()
	{
		// convolution filter size
		size_ = size_of(W_);
		
		// momentum
		V_ = volume_pool.get_double3d(size_);
		volume_utils::zero_out(V_);

		// weight L2-norm
		norm_ = volume_utils::square_sum(W_);
	}


public:
	conv_edge(base_node_ptr in, base_node_ptr out,
			  double3d_ptr W,
			  vec3i stride = vec3i::one,
			  vec3i sparse = vec3i::one,
			  double eta = 0.01,
			  double mom = 0,
			  double wc = 0, 
			  double szB = 1)
		: base_edge(in, out)
		, W_(W)
		, dEdW_()
		, V_()
		, fft_()
		, stride_(stride)
		, sparse_(vec3i::one)
		, norm_(0)
		, eta_(eta)
		, mom_(mom)
		, wc_(wc)
		, szB_(szB)

	{
		init();
		set_sparse(sparse);
	}

	virtual ~conv_edge() {}

}; // class conv_edge

typedef boost::shared_ptr<conv_edge> conv_edge_ptr;

}} // namespace zi::znn

#endif // ZNN_CONV_EDGE_HPP_INCLUDED
