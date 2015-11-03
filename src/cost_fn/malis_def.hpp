#ifndef ZNN_MALIS_DEF_HPP_INCLUDED
#define ZNN_MALIS_DEF_HPP_INCLUDED

#include "../core/types.hpp"

namespace zi {
namespace znn {

struct malis_weight
{
    std::list<double3d_ptr> merger;
    std::list<double3d_ptr> splitter;

    malis_weight(std::list<double3d_ptr> m, std::list<double3d_ptr> s)
        : merger(m)
        , splitter(s)
    {}
};

struct malis_metric
{
    double loss;

    double nTP;
    double nFP;
    double nFN;
    double nTN;

    malis_metric()
        : loss(0)
        , nTP(0)
        , nFP(0)
        , nFN(0)
        , nTN(0)
    {}
};

typedef std::pair<malis_weight, malis_metric> malis_return;

}} // namespace zi::znn

#endif // ZNN_MALIS_DEF_HPP_INCLUDED