#ifndef ZNN_MAKE_AFFINITY_HPP_INCLUDED
#define ZNN_MAKE_AFFINITY_HPP_INCLUDED

#include "../../core/types.hpp"
#include "../../core/utils.hpp"
#include "../../core/volume_utils.hpp"

namespace zi {
namespace znn {

inline std::list<double3d_ptr>
make_affinity( double3d_ptr vol_ptr, std::size_t dim = 3 )
{
    // only allows 2D or 3D affinity graph
    STRONG_ASSERT((dim==2)||(dim==3));

    vec3i s = size_of(vol_ptr);
    if ( s[2] > 1 ) dim = 3;

    double3d_ptr xaff = volume_pool.get_double3d(s);volume_utils::zero_out(xaff);
    double3d_ptr yaff = volume_pool.get_double3d(s);volume_utils::zero_out(yaff);
    double3d_ptr zaff;
    if ( dim == 3 )
    {
        zaff = volume_pool.get_double3d(s);
        volume_utils::zero_out(zaff);
    }

    double3d& vol = *vol_ptr;

    for ( std::size_t x = 0; x < s[0]; ++x )
        for ( std::size_t y = 0; y < s[1]; ++y )
            for ( std::size_t z = 0; z < s[2]; ++z )
            {
                if ( x > 0 )
                {
                    (*xaff)[x][y][z] = std::min(vol[x-1][y][z],vol[x][y][z]);
                }

                if ( y > 0 )
                {
                    (*yaff)[x][y][z] = std::min(vol[x][y-1][z],vol[x][y][z]);
                }

                if ( z > 0 )
                {
                    (*zaff)[x][y][z] = std::min(vol[x][y][z-1],vol[x][y][z]);
                }
            }

    std::list<double3d_ptr> ret;
    if ( dim == 3 ) ret.push_back(zaff);
    ret.push_back(yaff);
    ret.push_back(xaff);
    return ret;
}

}} // namespace zi::znn

#endif // ZNN_MAKE_AFFINITY_HPP_INCLUDED