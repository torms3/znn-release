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

#include "debug/malis_debug.hpp"
#include "debug/merger_debug.hpp"

#include <iostream>
#include <zi/time.hpp>
#include <zi/zargs/zargs.hpp>

ZiARG_string(options, "", "Option file path");
ZiARG_string(type, "", "Debug type");

using namespace zi::znn;

int main(int argc, char** argv)
{
    // options
    zi::parse_arguments(argc, argv);
    options_ptr op = options_ptr(new options(ZiARG_options));
    op->save();

    debug_ptr dbg;
    if ( ZiARG_type == "malis" )
    {
        dbg = debug_ptr(new malis_debug(op));
    }
    else if ( ZiARG_type == "merger" )
    {
        dbg = debug_ptr(new merger_debug(op));
    }

    if ( dbg ) dbg->run();
}