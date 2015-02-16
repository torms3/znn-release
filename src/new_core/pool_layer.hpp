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

#ifndef ZNN_POOL_LAYER_HPP_INCLUDED
#define ZNN_POOL_LAYER_HPP_INCLUDED

#include "layer.hpp"

namespace zi {
namespace znn {

template<class Manager>
class pool_layer : virtual public layer<Manager>
{
protected:
	layer_ptr 	top_;
	layer_ptr 	btm_;
};

}} // namespace zi::znn

#endif // ZNN_POOL_LAYER_HPP_INCLUDED
