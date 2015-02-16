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

#ifndef ZNN_WAITER_HPP_INCLUDED
#define ZNN_WAITER_HPP_INCLUDED

namespace zi {
namespace znn {

class waiter
{
private:
    std::size_t             remaining_;
    zi::mutex               mutex_;
    zi::condition_variable  cv_;

public:
    waiter()
        : remaining_(0)
    {}

    waiter(std::size_t how_many)
        : remaining_(how_many)
    {}

    void one_done()
    {
        zi::guard g(mutex_);
        --remaining_;
        if ( remaining_ == 0 )
        {
            cv_.notify_all();
        }
    }

    void wait()
    {
        zi::guard g(mutex_);
        while ( remaining_ )
        {
            cv_.wait(mutex_);
        }
    }

    void set(std::size_t how_many)
    {
        zi::guard g(mutex_);
        remaining_ = how_many;
        if ( remaining_ == 0 )
        {
            cv_.notify_all();
        }
    }

}; // class waiter;

}} // namespace zi::znn

#endif // ZNN_WAITER_HPP_INCLUDED
