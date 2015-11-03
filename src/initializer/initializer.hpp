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

#ifndef ZNN_INITIALIZER_HPP_INCLUDED
#define ZNN_INITIALIZER_HPP_INCLUDED

#include "../core/types.hpp"

#include <boost/random.hpp>

namespace zi {
namespace znn {

class initializer
{
public:
    virtual void initialize( double3d_ptr w ) = 0;
    virtual void init( const std::string& params ) = 0;

}; // abstract class initializer

typedef boost::shared_ptr<initializer> initializer_ptr;

// global random number generator
namespace {
boost::mt19937 rng = boost::mt19937(time(0));
} // anonymous namespace

// for generating thread-safe uniform[0,1] random numbers
struct zero_one_generator_impl
{
private:
  	typedef boost::uniform_real<> uniform_dist;
	typedef boost::variate_generator<boost::mt19937&, uniform_dist> generator;
    
    boost::mt19937 rng_;
    uniform_dist   dis_;
    generator      gen_;

    zi::mutex      m_;

public:
	zero_one_generator_impl()
		: rng_(time(0))
		, dis_(0,1)
		, gen_(rng_, dis_)
	{}

	double rand()
	{
		zi::mutex::guard g(m_);
		return gen_();
	}
}; // struct zero_one_generator_impl

// global thread-safe unit uniform random number generator
namespace {
zero_one_generator_impl& zero_one_generator = zi::singleton<zero_one_generator_impl>::instance();
} // anonymous namespace

}} // namespace zi::znn

#endif // ZNN_INITIALIZER_HPP_INCLUDED
