#ifndef ZNN_DEBUG_HPP_INCLUDED
#define ZNN_DEBUG_HPP_INCLUDED

#include "front_end/data_provider/data_providers.hpp"
#include "front_end/options.hpp"

namespace zi {
namespace znn {

class debug 
{
protected:
    data_provider_ptr   dp_;
    options_ptr         op_;


public:
    virtual void run() = 0;


public:
    debug( options_ptr op )
        : op_(op)
    {}

    virtual ~debug(){};

};  // class debug

typedef boost::shared_ptr<debug> debug_ptr;

}} // namespace zi::znn

#endif // ZNN_DEBUG_HPP_INCLUDED
