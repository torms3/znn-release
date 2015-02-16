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

#ifndef ZNN_BASE_EDGE_HPP_INCLUDED
#define ZNN_BASE_EDGE_HPP_INCLUDED

#include "../core/types.hpp"

namespace zi {
namespace znn {

class base_node;

typedef boost::shared_ptr<base_node> base_node_ptr;

class base_edge
{
protected:
    zi::mutex       m_;

protected:
    base_node_ptr   in_ ;
    base_node_ptr   out_;

protected:
	vec3i			size_;


public:
	vec3i size() const
	{
		return size_;
	}

	virtual vec3i real_size() const
	{
		return size_;
	}

	base_node_ptr in()
	{
		return in_;
	}

	base_node_ptr out()
	{
		return out_;
	}

public:
	virtual void forward()  = 0;
	virtual void backward() = 0;
	virtual void update()	= 0;


public:
	base_edge(base_node_ptr in, base_node_ptr out)
		: m_()
		, in_(in)
		, out_(out)
		, size_(vec3i::one)
	{}

	virtual ~base_edge() {};

}; // abstract class base_edge

typedef boost::shared_ptr<base_edge> base_edge_ptr;

}} // namespace zi::znn

#endif // ZNN_BASE_EDGE_HPP_INCLUDED
