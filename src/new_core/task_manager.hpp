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

#ifndef ZNN_TASK_MANAGER_HPP_INCLUDED
#define ZNN_TASK_MANAGER_HPP_INCLUDED

#include <zi/concurrency.hpp>
#include <zi/utility/singleton.hpp>

namespace zi {
namespace znn {

class task_manager_impl
{
private:
	typedef zi::task_manager::prioritized  			task_manager_type;
	typedef boost::shared_ptr<task_manager_type> 	task_manager_ptr;

private:
	task_manager_ptr	tm_;


public:
	void operator()(std::size_t worker_limit)
	{
		if ( tm_ )
		{
			tm_->join();
			tm_.reset();
		}

		tm_ = task_manager_ptr(new task_manager_type(worker_limit));	
	}

	task_manager_type& operator()()
	{
		return *tm_;
	}

public:
	task_manager_impl() : tm_() {}
	~task_manager_impl() { tm_->join(); }

}; // class task_manager

namespace {
task_manager_impl& task_manager =
	zi::singleton<task_manager_impl>::instance();
} // anonymous namespace

}} // namespace zi::znn

#endif // ZNN_TASK_MANAGER_HPP_INCLUDED