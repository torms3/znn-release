//
// Copyright (C) 2014  Kisuk Lee           <kisuklee@mit.edu>
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

#include "front_end/options.hpp"

#include "debug/get_segmentation_debug.hpp"
#include "debug/malis_2d_debug.hpp"

#include <iostream>
#include <zi/time.hpp>
#include <zi/zargs/zargs.hpp>

ZiARG_string(options, "", "Option file path");
ZiARG_string(type, "", "Debug type");
ZiARG_double(high, 1, "High threshold");
ZiARG_double(low, 0, "Low threshold");

using namespace zi::znn;

int main(int argc, char** argv)
{
    // options
    zi::parse_arguments(argc, argv);
    options_ptr op = options_ptr(new options(ZiARG_options,false));

    debug_ptr dbg;
    if ( ZiARG_type == "malis_2d" )
    {
        dbg = debug_ptr(new malis_2d_debug(op,ZiARG_high,ZiARG_low));
    }
    else if ( ZiARG_type == "get_segmentation" )
    {
        dbg = debug_ptr(new get_segmentation_debug(op));
    }
    else
    {
        STRONG_ASSERT(false);
    }

    if ( dbg ) dbg->run();
}