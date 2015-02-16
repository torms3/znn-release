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

#ifndef ZNN_LAYER_HPP_INCLUDED
#define ZNN_LAYER_HPP_INCLUDED

#include "../core/types.hpp"

#include <vector>

namespace zi {
namespace znn {

template<class Manager, typename Node>
class layer
{
protected:
	Manager	tm_;

protected:
	std::vector<Node>	nodes_;


public:
	virtual void run_forward(std::size_t n)
	{
		FOR_EACH( it, nodes_ )
		{
			tm_->insert(
					zi::bind(),
					priority
				);
		}
	}

protected:
	virtual void forward_edge()
	{
		
	}

public:
	virtual ~layer() = 0;
};

typedef boost::shared_ptr<layer> layer_ptr;

template<class Manager>
class async_layer : virtual public layer<Manager>
{
protected:
	layer_ptr	top_;
	layer_ptr	btm_;
};

template<class Manager>
class sync_layer : virtual public layer<Manager>
{
protected:
	layer_ptr	top_;
	layer_ptr	btm_;
};

}} // namespace zi::znn

#endif // ZNN_LAYER_HPP_INCLUDED
