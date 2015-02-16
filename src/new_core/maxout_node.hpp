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

#ifndef ZNN_MAXOUT_NODE_HPP_INCLUDED
#define ZNN_MAXOUT_NODE_HPP_INCLUDED

#include "base_node.hpp"

namespace zi {
namespace znn {

class maxout_node : virtual public base_node
{
protected:
	long3d_ptr maxidx_;


public:
    long3d_ptr& maxout_indices()
    {
        return maxidx_;
    }


public:
    maxout_node(const std::string& name,
                std::size_t layer_no = 0,
                std::size_t neuron_no = 0)
        : base_node(name, layer_no, neuron_no)
        , maxidx_()
    {}

    virtual ~maxout_node() {};

}; // class maxout_node

typedef boost::shared_ptr<maxout_node> maxout_node_ptr;

}} // namespace zi::znn

#endif // ZNN_MAXOUT_NODE_HPP_INCLUDED
