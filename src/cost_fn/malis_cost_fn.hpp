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

#ifndef ZNN_MALIS_COST_FN_HPP_INCLUDED
#define ZNN_MALIS_COST_FN_HPP_INCLUDED

#include "cost_fn.hpp"
#include "../core/volume_pool.hpp"

namespace zi {
namespace znn {

class malis_cost_fn: virtual public cost_fn
{
public:
    virtual double3d_ptr gradient( double3d_ptr output,
                                   double3d_ptr label,
                                   bool3d_ptr   mask )
    {
    }

    virtual std::list<double3d_ptr> gradient( std::list<double3d_ptr> outputs,
                                              std::list<double3d_ptr> labels,
                                              std::list<bool3d_ptr>   masks )
    {
    }

    virtual double compute_cost( double3d_ptr output,
                                 double3d_ptr label,
                                 bool3d_ptr   mask )
    {
    }

    virtual double compute_cost( std::list<double3d_ptr> outputs, 
                                 std::list<double3d_ptr> labels,
                                 std::list<bool3d_ptr>   masks )
    {
    }

    virtual void print_cost( double cost )
    {
        std::cout << "SQSQ (m=" << margin_ << "): " << cost;
    }


public:
    malis_cost_fn()
    {}

}; // class malis_cost_fn

}} // namespace zi::znn

#endif // ZNN_MALIS_COST_FN_HPP_INCLUDED
