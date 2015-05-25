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

#ifndef ZNN_VOLUME_2D_DATA_PROVIDER_HPP_INCLUDED
#define ZNN_VOLUME_2D_DATA_PROVIDER_HPP_INCLUDED

#include "volume_data_provider.hpp"

namespace zi {
namespace znn {

class volume_2d_data_provider : virtual public volume_2d_data_provider
{

// constructor & destructor
public:
    virtual ~volume_2d_data_provider()
    {}

}; // class volume_2d_data_provider

}} // namespace zi::znn

#endif // ZNN_VOLUME_2D_DATA_PROVIDER_HPP_INCLUDED