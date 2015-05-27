//
// Copyright (C) 2015  Kisuk Lee <kisuklee@mit.edu>
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

#ifndef ZNN_FEATURE_MAP_SCANNER_HPP_INCLUDED
#define ZNN_FEATURE_MAP_SCANNER_HPP_INCLUDED

#include "../net.hpp"
#include "../data_spec/rw_volume_data.hpp"

#include <map>

namespace zi {
namespace znn {

class feature_map_scanner
{
private:
	typedef rw_dvolume_data 	fmap_type;
	typedef rw_dvolume_data_ptr fmap_type_ptr;

private:
	std::map<std::string, std::vector<fmap_type_ptr> > fmaps_;

	net_ptr			net_;

	const vec3i		uc_;
	const vec3i		lc_;

	bool 			all_;


public:
	void scan(vec3i loc)
	{
		FOR_EACH( it, net_->node_groups_ )
		{
			std::vector<std::size_t> scan_list;
			if ( all_ )
			{
				std::size_t n = (*it)->count();
				for ( std::size_t i = 0; i < n; ++i )
				{
					scan_list.push_back(i+1);
				}
			}
			else
			{
				scan_list = (*it)->spec()->scan_list;
			}

			std::size_t n = scan_list.size();
			for ( std::size_t i = 0; i < n; ++i )
			{
				node_ptr nd = (*it)->nodes_[scan_list[i]-1];
				push_feature_map(loc, (*it)->name(), nd, i);
			}
		}
	}

	void save_map(const std::string& fpath)
	{	
		FOR_EACH( it, net_->node_groups_ )
		{
			std::vector<std::size_t> scan_list;
			if ( all_ )
			{
				std::size_t n = (*it)->count();
				for ( std::size_t i = 0; i < n; ++i )
				{
					scan_list.push_back(i+1);
				}
			}
			else
			{
				scan_list = (*it)->spec()->scan_list;
			}

			std::size_t n = scan_list.size();
			for ( std::size_t i = 0; i < n; ++i )
			{
				node_ptr nd = (*it)->nodes_[scan_list[i]-1];
				save_feature_map(fpath, (*it)->name(), nd, i);
			}
		}
	}

	void save_tensor(const std::string& fpath)
	{
		FOR_EACH( it, fmaps_ )
		{
			std::string  name = it->first;
			std::string fname = fpath + name;
			
			std::vector<double3d_ptr> tensor;
			FOR_EACH( jt, it->second )
			{
				tensor.push_back((*jt)->get_volume());
			}

			volume_utils::save_tensor(tensor, fname);
		}
	}

private:
	void push_feature_map(vec3i loc, std::string layer, node_ptr nd, std::size_t idx)
	{
		fmaps_[layer][idx]->set_patch(loc, nd->get_activation());
	}

	void save_feature_map(const std::string& fpath, std::string layer, node_ptr nd, std::size_t idx)
	{
		std::string num = boost::lexical_cast<std::string>(nd->get_neuron_number());
		std::string fname = fpath + layer + ".map" + num;
		double3d_ptr fmap = fmaps_[layer][idx]->get_volume();
		volume_utils::save(fmap, fname);
		export_size_info(size_of(fmap), fname);
	}


private:
	void initialize()
	{
		FOR_EACH( it, net_->node_groups_ )
		{
			if ( all_ )
			{
				FOR_EACH( jt, (*it)->nodes_ )
				{
					add_feature_map(*it, *jt);
				}
			}
			else
			{
				FOR_EACH( jt, (*it)->spec()->scan_list )
				{
					std::size_t no = *jt;
					STRONG_ASSERT(no > 0);
					STRONG_ASSERT(no <= (*it)->count());
					node_ptr nd = (*it)->nodes_[no-1];
					add_feature_map(*it, nd);	
				}
			}
		}
	}

	void add_feature_map(node_group_ptr ng, node_ptr nd)
	{
		vec3i sz = nd->get_filtered_size();
		box a = box::centered_box(uc_,sz);
		box b = box::centered_box(lc_,sz);
		box c = a + b;
			
		double3d_ptr vol = volume_pool.get_double3d(c.size());
		vec3i offset = c.upper_corner();
		vec3i FoV = sz;
		fmap_type* fmap  = new fmap_type(vol,FoV,offset);

		// std::cout << "Layer " << ng->name() << ", Map " << nd->get_neuron_number();
		fmaps_[ng->name()].push_back(fmap_type_ptr(fmap));
		// STRONG_ASSERT(fmaps_[ng->name()].size()==nd->get_neuron_number());
	}


public:
	feature_map_scanner(net_ptr net, vec3i uc, vec3i lc, bool all = false)
		: net_(net)
		, uc_(uc)
		, lc_(lc)
		, all_(all)
	{
		initialize();
	}

}; // class feature_map_scanner

typedef boost::shared_ptr<feature_map_scanner> feature_map_scanner_ptr;

}} // namespace zi::znn

#endif // ZNN_FEATURE_MAP_SCANNER_HPP_INCLUDED