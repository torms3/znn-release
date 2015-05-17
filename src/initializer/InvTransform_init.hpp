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

#ifndef ZNN_INVTRANSFORM_INIT_HPP_INCLUDED
#define ZNN_INVTRANSFORM_INIT_HPP_INCLUDED

#include "Transform_init.hpp"

namespace zi {
namespace znn {

class InvTransform_init : virtual public Transform_init
{
public:    
    virtual void initialize( double3d_ptr w )
    {
    	Transform_init::initialize(w);
    	
    	// Invert
    	std::cout << "Invert from [" << lower << "," << upper << "] ";

    	std::size_t n = w->num_elements();    	
	    for ( std::size_t i = 0; i < n; ++i )
	    {
	        w->data()[i] = upper - w->data()[i] + lower;
	    }

	    std::cout << "to [" << upper << "," << lower << "]" << std::endl;
    }
    
public:
	InvTransform_init( double _lower = static_cast<double>(0),
				 	   double _upper = static_cast<double>(1) )
		: Transform_init(_lower,_upper)
	{}

}; // class InvTransform_init

}} // namespace zi::znn

#endif // ZNN_INVTRANSFORM_INIT_HPP_INCLUDED