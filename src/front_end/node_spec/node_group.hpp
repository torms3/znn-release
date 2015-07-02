//
// Copyright (C) 2014  Kisuk Lee <kisuklee@mit.edu>
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

#ifndef ZNN_NODE_GROUP_HPP_INCLUDED
#define ZNN_NODE_GROUP_HPP_INCLUDED

#include "../../core/node.hpp"
#include "node_spec.hpp"
#include "../edge_spec/edge_group.hpp"

namespace zi {
namespace znn {

typedef std::list<edge_group_ptr> connections;

class node_group
{
private:	
	const std::string		name_;
	node_spec_ptr			spec_;
	std::vector<node_ptr>	nodes_;

	connections 			in_;
	connections 			out_;

	// initialization flags
	bool					sparse_init_;
	bool					layer_init_;
	bool					size_init_;

	// layer
	std::size_t				layer_;

	// sizes
	vec3i					in_size_;
	vec3i					out_size_;
	vec3i					in_sparse_;
	vec3i					out_sparse_;

	// bias loaded
	bool					loaded_;

	// cost function
	cost_fn_ptr				cost_fn_;

	// crop
	bool					crop_;


public:
	std::list<double3d_ptr> get_activations( std::size_t n, bool softmax = false )
	{
		std::list<double3d_ptr> activations;
		FOR_EACH( it, nodes_ )
		{
			activations.push_back((*it)->wait(n));
		}
		if ( softmax )
		{
			activations = volume_utils::softmax(activations);
		}
		return activations;
	}


public:
	void set_learning_rate( double eta )
	{
		spec_->eta = eta;
		FOR_EACH( it, nodes_ )
		{
			(*it)->set_eta(eta);
		}
	}

	void set_momentum( double mom )
	{
		spec_->mom = mom;
		FOR_EACH( it, nodes_ )
		{
			(*it)->set_momentum(mom);
		}
	}

	void set_weight_decay( double wc )
	{
		spec_->wc = wc;
		FOR_EACH( it, nodes_ )
		{
			(*it)->set_weight_decay(wc);
		}
	}

	void set_minibatch_size( double sz )
	{
		FOR_EACH( it, nodes_ )
		{
			(*it)->set_minibatch_size(sz);
		}
	}

	void sends_fft( bool b )
	{
		FOR_EACH( it, nodes_ )
		{
			(*it)->sends_fft(b);
		}
	}

	void receives_fft( bool b )
	{
		spec_->fft = b;

		FOR_EACH( it, nodes_ )
		{
			(*it)->receives_fft(b);
		}
	}

	void set_fft_profile()
	{
		receives_fft(spec_->fft);
	}

	void disable_filtering()
	{
		STRONG_ASSERT( !initialized() );
		spec_->filter_size 	 = vec3i::one;
		spec_->filter_stride = vec3i::one;

		FOR_EACH( it, nodes_ )
		{
			(*it)->set_filtering(std::greater<double>(),vec3i::one);
		}
	}

	void set_crop( bool b )
	{
		crop_ = b;
	}

	bool is_crop() const
	{
		return crop_;
	}

	
// only allow node/edge factories to modify node_group
// by declaring node/edge factories as friends
private:
	void set_spec( node_spec_ptr spec )
	{
		spec_ = spec;
	}

	void add_node( node_ptr node )
	{		
		nodes_.push_back(node);
	}

	void add_in_connection( edge_group_ptr e )
	{
		in_.push_back(e);
	}

	void add_out_connection( edge_group_ptr e )
	{
		out_.push_back(e);
	}


public:
	void forward_sparse( const vec3i& sparse = vec3i::one )
	{
		if ( initialized() )
		{
			STRONG_ASSERT(in_sparse_ == sparse);
		}
		else
		{
			in_sparse_ = sparse;
			if ( spec_->sparse != vec3i::zero )
			{
				out_sparse_ = spec_->sparse;
			}
			else
			{
				out_sparse_ = sparse*spec_->filter_stride;
			}
			init_sparse();
		}

		FOR_EACH( it, out_ )
		{
			(*it)->set_sparse(out_sparse_);
			(*it)->target_->forward_sparse(out_sparse_);
		}
	}

	void forward_layer_number( std::size_t layer = 0 )
	{
		if ( initialized() )
		{
			STRONG_ASSERT(layer_ >= layer );
		}
		else
		{
			layer_ = std::max(layer_,layer);
			init_layers();
		}

		FOR_EACH( it, out_ )
		{
			(*it)->target_->forward_layer_number(layer+1);
		}
	}

	void forward_init( const vec3i& size, 
					   const vec3i& sparse = vec3i::one,
					   std::size_t layer = 0 )
	{
		if ( initialized() )
		{
			STRONG_ASSERT(in_size_ == size);
			STRONG_ASSERT(in_sparse_ == sparse);
			STRONG_ASSERT(layer_ >= layer);
		}
		else
		{
			in_size_ = size;
			in_sparse_ = sparse;
			out_size_ = size - spec_->real_filter_size(sparse) + vec3i::one;
			out_sparse_ = sparse*spec_->filter_stride;
			layer_ = std::max(layer_,layer);

			init_sparse();
			init_layers();
			init_sizes();
		}

		FOR_EACH( it, out_ )
		{
			(*it)->set_sparse(out_sparse_);
			vec3i new_size = out_size_ - (*it)->real_filter_size() + vec3i::one;
			(*it)->target_->forward_init(new_size,out_sparse_,layer+1);
		}
	}

	void backward_init( const vec3i& size = vec3i::one )
	{
		if ( initialized() )
		{
			if ( out_size_ != size )
			{
				display();

				std::cout << "out_size_ = " << out_size_ << std::endl;
				std::cout << "size = " << size << std::endl;

				STRONG_ASSERT(false);
			}
		}
		else
		{
			out_size_ = size;
			in_size_ = size + spec_->real_filter_size(in_sparse_) - vec3i::one;
			init_sizes();
		}

		FOR_EACH( it, in_ )
		{
			// (*it)->set_sparse(in_sparse_);
			vec3i new_size = in_size_;
			if ( (*it)->is_crop() )
			{
				vec3i crop_offset = (*it)->spec()->crop_offset;
				new_size += (crop_offset + crop_offset);
			}
			else
			{
				new_size += ((*it)->real_filter_size() - vec3i::one);
			}
			(*it)->source_->backward_init(new_size);
		}
	}

private:
	bool initialized() const
	{
		return sparse_init_ & layer_init_ & size_init_;
	}

	void init_sparse()
	{
		FOR_EACH( it, nodes_ )
		{
			(*it)->set_sparse(in_sparse_);
		}
		sparse_init_ = true;
	}

	void init_layers()
	{
		FOR_EACH( it, nodes_ )
		{
			(*it)->set_layer_number(layer_);
		}
		layer_init_ = true;
	}

	void init_sizes()
	{
		FOR_EACH( it, nodes_ )
		{
			(*it)->set_size(in_size_);			
			(*it)->set_step_size(spec_->filter_stride);
		}
		size_init_ = true;
	}


public:
	std::size_t count() const
	{
		return nodes_.size();
	}

	std::string name() const
	{
		return name_;
	}

	node_spec_ptr spec()
	{
		return spec_;
	}

	std::size_t count_in_connections() const
	{
		return in_.size();
	}

	std::size_t count_out_connections() const
	{
		return out_.size();
	}

	std::size_t layer() const
	{
		return layer_;
	}

	void test_print() const
	{
		std::cout << "[" << name_ << ": leyer " << layer_ << "]" << std::endl;
	}

	double fan_in() const
	{
		return nodes_.front()->fan_in();
	}

	double fan_out() const
	{
		return nodes_.front()->fan_out();
	}

	bool is_loaded() const
	{
		return loaded_;
	}

	const vec3i& get_in_size() const
	{
		return in_size_;
	}


public:
	void save( const std::string& path, bool history = false ) const
	{
		spec_->save(path);	// save node specification
		if ( crop_ ) return;
		save_bias(path); 	// save bias

		if ( history )
		{
			accumulate_weight(path);
		}
	}

private:
	#define BINARY_WRITE (std::ios::out | std::ios::binary)
	#define BINARY_READ	 (std::ios::in | std::ios::binary)
	#define BINARY_ACCUM (BINARY_WRITE | std::ios::app)

	void save_bias( const std::string& path ) const
	{
		if ( crop_ ) return;

		std::string fpath = path + name_ + ".weight";
        std::ofstream fout(fpath.c_str(), BINARY_WRITE);

        FOR_EACH( it, nodes_ )
        {
        	(*it)->print_bias(fout);
        }

        fout.close();
	}

	void accumulate_weight( const std::string& path ) const
	{
		if ( crop_ ) return;

		std::string fpath = path + name_ + ".weight.hist";
        std::ofstream fout(fpath.c_str(), BINARY_ACCUM);

        FOR_EACH( it, nodes_ )
        {
        	(*it)->print_bias(fout);
        }
        
        fout.close();
	}

	bool load_spec( const std::string& path )
	{
		std::string fpath = path + name_ + ".spec";
		return spec_->build(fpath);
	}

	bool load_bias( std::ifstream& fin )
	{
		if ( crop_ ) return false;

		STRONG_ASSERT( fin );

		FOR_EACH( it, nodes_ )
		{
			double bias;
			fin.read(reinterpret_cast<char*>(&bias),sizeof(bias));
			(*it)->set_bias(bias);
			// std::cout << "node " << (*it)->get_neuron_number() 
			// 		  << ": bias = " << bias << std::endl;
		}

		fin.close();
		loaded_ = true;
		return true;
	}

	bool load_bias( const std::string& path )
	{
		if ( crop_ ) return false;

		std::string fpath = path + name_ + ".weight";
		std::ifstream fin(fpath.c_str(), BINARY_READ);
		if ( !fin )
		{
			fpath = path + name_ + ".bias";
			fin.clear();fin.open(fpath.c_str(), BINARY_READ);
			if ( !fin ) return false;
		}

		return load_bias(fin);
	}

	bool load_bias( const std::string& path, std::size_t idx )
	{
		if ( crop_ ) return false;

		std::string fpath = path + name_ + ".weight.hist";
		std::ifstream fin(fpath.c_str(), BINARY_READ);
		if ( !fin ) return false;

		// get length of the file
		fin.seekg(0,fin.end);
		int len = fin.tellg();
		STRONG_ASSERT( len > 0 );
		
		// idx validity check
		std::size_t sz = sizeof(double)*count();
		STRONG_ASSERT( len % sz == 0 );
		STRONG_ASSERT( len/sz > idx );

		// seek
		int pos = sz*idx;
		fin.seekg(pos,fin.beg);

		return load_bias(fin);
	}


public:
	void display(std::ostream& os = std::cout)
	{
		os << "[" << name() << "]" << "\n";
		os << "Node size:\t\t" << nodes_.front()->get_size()
		   << " x " << count() << "\n";

		if ( !spec()->scan_list.empty() )
		{
			print_range("Scan list",spec()->scan_list);
		}
	  	
		if ( crop_ ) return;

	  	// filter
		if ( spec_->filter_size != vec3i::one )
		{
		  	os << "Filter size:\t\t" << spec_->filter_size << "\n";
		  	os << "Filter stride:\t\t" << spec_->filter_stride << "\n";
		  	
		  	vec3i sparse = nodes_.front()->get_sparse();
		  	if ( sparse != vec3i::one )
		  	{
		  		os << "Sparseness:\t\t" << sparse << "\n";
		  		os << "Real filter size:\t" 
		  				  << spec_->real_filter_size(sparse) << "\n";
		  	}
		}

		// FFT
		if ( count_in_connections() )
		{
			os << "Receive FFT: " << spec_->fft << "\n";
		}
	}


// Comparison
public:
	inline bool
	operator<( const node_group& rhs )
	{
		return layer_ < rhs.layer_;
	}	

	inline bool
	operator>( const node_group& rhs )
	{
		return layer_ > rhs.layer_;
	}

	inline bool
	operator<=( const node_group& rhs )
	{
		return !(*this > rhs);
	}

	inline bool
	operator>=( const node_group& rhs )
	{
		return !(*this < rhs);
	}


// only allow node factory to create node_group
// by declaring node factory as a friend
private:	
	node_group( const std::string& name )
		: name_(name)
		, spec_()
		, nodes_()
		, in_()
		, out_()
		, sparse_init_(false)
		, layer_init_(false)
		, size_init_(false)
		, layer_(0)
		, in_size_(vec3i::zero)
		, out_size_(vec3i::zero)
		, in_sparse_(vec3i::one)
		, out_sparse_(vec3i::one)
		, loaded_(false)
		, crop_(false)
	{
		set_spec(node_spec_ptr(new node_spec(name_)));	// default spec
		// std::cout << "node_group " << name_ << " has created!" << std::endl;
	}

	friend class node_factory_impl;
	friend class edge_factory_impl;
	friend class neuron_group;
	friend class net;
	friend class feature_map_scanner;

}; // class node_group

typedef boost::shared_ptr<node_group> node_group_ptr;

}} // namespace zi::znn

#endif // ZNN_NODE_GROUP_HPP_INCLUDED
