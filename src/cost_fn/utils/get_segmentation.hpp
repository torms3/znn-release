#ifndef ZNN_GET_SEGMENTATION_HPP_INCLUDED
#define ZNN_GET_SEGMENTATION_HPP_INCLUDED

#include "../../core/types.hpp"
#include "../../core/utils.hpp"
#include "../../core/volume_utils.hpp"

#include <zi/disjoint_sets/disjoint_sets.hpp>

namespace zi {
namespace znn {

inline long3d_ptr
get_segmentation( std::list<double3d_ptr> affs,
                  double threshold = 0.5 )
{
    vec3i s = volume_utils::volume_size(affs.front());

    double3d_ptr xaff = affs.front(); affs.pop_front();
    double3d_ptr yaff = affs.front(); affs.pop_front();;
    double3d_ptr zaff;
    if ( s[2] > 1 )
    {
        zaff = affs.front();
        affs.pop_front();
    }

    std::size_t n = s[0]*s[1]*s[2];

    zi::disjoint_sets<uint32_t> sets(n+1);
    std::vector<uint32_t>       sizes(n+1);

    long3d_ptr ids_ptr = volume_pool.get_long3d(s);
    long3d& ids = *ids_ptr;

    for ( std::size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
    }

    typedef std::pair<uint32_t,uint32_t> edge_type;
    std::vector<edge_type> edges;

    // thresholded affinity graph
    for ( std::size_t z = 0; z < s[2]; ++z )
    {
        for ( std::size_t y = 0; y < s[1]; ++y )
        {
            for ( std::size_t x = 0; x < s[0]; ++x )
            {
                long id1 = ids[x][y][z];

                if ( x > 0 )
                {
                    // skip disconnected (black) edges
                    // only count connected (white) edges
                    if ( (*xaff)[x][y][z] > threshold )
                    {
                        long id2 = ids[x-1][y][z];
                        edges.push_back(edge_type(id1, id2));
                    }
                }

                if ( y > 0 )
                {
                    // skip disconnected (black) edges
                    // only count connected (white) edges
                    if ( (*yaff)[x][y][z] > threshold )
                    {
                        long id2 = ids[x][y-1][z];
                        edges.push_back(edge_type(id1, id2));
                    }
                }

                if ( z > 0 )
                {
                    // skip disconnected (black) edges
                    // only count connected (white) edges
                    if ( (*yaff)[x][y][z] > threshold )
                    {
                        long id2 = ids[x][y][z-1];
                        edges.push_back(edge_type(id1, id2));
                    }
                }
            }
        }
    }

    // connected components of the thresholded affinity graph
    // computed the size of each connected component
    FOR_EACH( it, edges )
    {
        uint32_t set1 = sets.find_set(it->first);
        uint32_t set2 = sets.find_set(it->second);

        if ( set1 != set2 )
        {
            uint32_t new_set = sets.join(set1, set2);

            sizes[set1] += sizes[set2];
            sizes[set2]  = 0;

            std::swap(sizes[new_set], sizes[set1]);
        }
    }

    std::vector<uint32_t> remaps(n+1);

    uint32_t next_id = 1;

    // assign a unique segment ID to each of
    // the pixel in a connected component
    for ( std::size_t i = 0; i < n; ++i )
    {
        uint32_t id = sets.find_set(ids.data()[i]);
        if ( sizes[id] > 1 )
        {
            if ( remaps[id] == 0 )
            {
                remaps[id] = next_id;
                ids.data()[i] = next_id;
                ++next_id;
            }
            else
            {
                ids.data()[i] = remaps[id];
            }
        }
        else
        {
            remaps[id] = 0;
            ids.data()[i] = 0;
        }
    }

    return ids_ptr;
}

}} // namespace zi::znn

#endif // ZNN_GET_SEGMENTATION_HPP_INCLUDED