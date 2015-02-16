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

namespace zi {
namespace znn {

class feature_map_scanner
{
private:
	std::map<std::string,rw_dvolume_data_ptr> fmaps_;

	net_ptr			net_;

	const vec3i		uc_;
	const vec3i		lc_;


public:
	void scan(vec3i loc)
	{
		// traverse every node
		FOR_EACH( it, net_->nodes_ )
		{
			// skip input and output nodes
			if ( (*it)->count_in_edges()  == 0 ||
				 (*it)->count_out_edges() == 0 )
				continue;

			push_feature_map(loc, *it);
		}
	}

	void save(const std::string& fpath)
	{
		// traverse every node
		FOR_EACH( it, net_->nodes_ )
		{
			// skip input and output nodes
			if ( (*it)->count_in_edges()  == 0 ||
				 (*it)->count_out_edges() == 0 )
				continue;

			save_feature_map(fpath, (*it)->get_name());
		}
	}

private:
	void push_feature_map(vec3i loc, node_ptr nd)
	{
		fmaps_[nd->get_name()]->set_patch(loc, nd->get_activation());
	}

	void save_feature_map(const std::string& fpath, const std::string& name)
	{
		std::string fname = fpath + name;
		double3d_ptr fmap = fmaps_[name]->get_volume();
		volume_utils::save(fmap,fname);
		export_size_info(size_of(fmap),fname);
	}


private:
	void initialize()
	{
		// traverse every node
		FOR_EACH( it, net_->nodes_ )
		{
			// skip input and output nodes
			if ( (*it)->count_in_edges()  == 0 ||
				 (*it)->count_out_edges() == 0 )
				continue;

			add_feature_map(*it);
		}
	}

	void add_feature_map(node_ptr nd)
	{
		vec3i sz = nd->get_filtered_size();
		box a = box::centered_box(uc_,sz);
		box b = box::centered_box(lc_,sz);
		box c = a + b;
			
		double3d_ptr vol = volume_pool.get_double3d(c.size());
		vec3i offset = c.upper_corner();
		vec3i FoV = sz;
		rw_dvolume_data* fmap  = new rw_dvolume_data(vol,FoV,offset);

		fmaps_[nd->get_name()] = rw_dvolume_data_ptr(fmap);
	}


public:
	feature_map_scanner(net_ptr net, vec3i uc, vec3i lc)
		: net_(net)
		, uc_(uc)
		, lc_(lc)
	{
		initialize();
	}

}; // class feature_map_scanner

typedef boost::shared_ptr<feature_map_scanner> feature_map_scanner_ptr;

}} // namespace zi::znn

#endif // ZNN_FEATURE_MAP_SCANNER_HPP_INCLUDED