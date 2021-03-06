#ifndef ZNN_MALIS_HPP_INCLUDED
#define ZNN_MALIS_HPP_INCLUDED

#include "../core/types.hpp"
#include "../core/utils.hpp"
#include "../core/volume_utils.hpp"

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <zi/disjoint_sets/disjoint_sets.hpp>
#include <zi/utility/for_each.hpp>
#include <vector>
#include <functional>
#include <algorithm>

namespace zi {
namespace znn {

inline long3d_ptr
get_segmentation( std::list<double3d_ptr> affs,
                  std::list<bool3d_ptr>   masks,
                  double threshold = 0.5 )
{
    vec3i s = volume_utils::volume_size(affs.front());

    double3d_ptr xaff = affs.front(); affs.pop_front();
    double3d_ptr yaff = affs.front(); affs.pop_front();
    double3d_ptr zaff = affs.front(); affs.pop_front();

    bool3d_ptr xmask = masks.front(); masks.pop_front();
    bool3d_ptr ymask = masks.front(); masks.pop_front();
    bool3d_ptr zmask = masks.front(); masks.pop_front();

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
    for ( std::size_t x = 0; x < s[0]; ++x )
        for ( std::size_t y = 0; y < s[1]; ++y )
            for ( std::size_t z = 0; z < s[2]; ++z )
            {
                long id1 = ids[x][y][z];

                if ( x > 0 )
                {
                    if ( (*xmask)[x][y][z] && ((*xaff)[x][y][z] > threshold) )
                    {
                        long id2 = ids[x-1][y][z];
                        edges.push_back(edge_type(id1, id2));
                    }
                }

                if ( y > 0 )
                {
                    if ( (*ymask)[x][y][z] && ( (*yaff)[x][y][z] > threshold) )
                    {
                        long id2 = ids[x][y-1][z];
                        edges.push_back(edge_type(id1, id2));
                    }
                }

                if ( z > 0 )
                {
                    if ( (*zmask)[x][y][z] && ( (*zaff)[x][y][z] > threshold ) )
                    {
                        long id2 = ids[x][y][z-1];
                        edges.push_back(edge_type(id1, id2));
                    }
                }
            }

    // connected components of the thresholded affinity graph
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



inline std::pair<std::list<double3d_ptr>, double>
malis( std::list<double3d_ptr> true_affs,
       std::list<double3d_ptr> affs,
       std::list<bool3d_ptr> masks,
       double margin)
{
    double loss = 0;
    long3d_ptr seg_ptr = get_segmentation(true_affs, masks);
    long3d&    seg = *seg_ptr;

    ZI_ASSERT(affs.size()==3);

    vec3i s = volume_utils::volume_size(affs.front());
    std::size_t n = s[0]*s[1]*s[2];

    double3d_ptr xaff = affs.front(); affs.pop_front();
    double3d_ptr yaff = affs.front(); affs.pop_front();
    double3d_ptr zaff = affs.front(); affs.pop_front();

    bool3d_ptr xmask = masks.front(); masks.pop_front();
    bool3d_ptr ymask = masks.front(); masks.pop_front();
    bool3d_ptr zmask = masks.front(); masks.pop_front();

    double3d_ptr xres = volume_pool.get_double3d(s); volume_utils::zero_out(xres);
    double3d_ptr yres = volume_pool.get_double3d(s); volume_utils::zero_out(yres);
    double3d_ptr zres = volume_pool.get_double3d(s); volume_utils::zero_out(zres);

    affs.push_back(xres);
    affs.push_back(yres);
    affs.push_back(zres);

    zi::disjoint_sets<uint32_t> sets(n+1);
    std::vector<uint32_t>       sizes(n+1);
    std::vector<std::map<uint32_t, uint32_t> > contains(n+1);


    long3d_ptr ids_ptr = volume_pool.get_long3d(s);
    long3d& ids = *ids_ptr;

    for ( std::size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
        contains[i+1][seg.data()[i]] = 1;
    }

    std::size_t n_pairs = n*(n-1)/2;

    typedef boost::tuple<double,uint32_t,uint32_t,double*> edge_type;
    typedef std::greater<edge_type>                        edge_compare;

    std::vector<edge_type> edges;
    edges.reserve(n*3);

    for ( std::size_t x = 0; x < s[0]; ++x )
        for ( std::size_t y = 0; y < s[1]; ++y )
            for ( std::size_t z = 0; z < s[2]; ++z )
            {
                if ( x > 0 )
                {
                    if ( (*xmask)[x][y][z] )
                        edges.push_back(edge_type((*xaff)[x][y][z], ids[x-1][y][z],
                                                  ids[x][y][z], &((*xres)[x][y][z])));
                }

                if ( y > 0 )
                {
                    if ( (*ymask)[x][y][z] )
                        edges.push_back(edge_type((*yaff)[x][y][z], ids[x][y-1][z],
                                                  ids[x][y][z], &((*yres)[x][y][z])));
                }

                if ( z > 0 )
                {
                    if ( (*zmask)[x][y][z] )
                        edges.push_back(edge_type((*zaff)[x][y][z], ids[x][y][z-1],
                                                  ids[x][y][z], &((*zres)[x][y][z])));
                }
            }

    std::sort(edges.begin(), edges.end(), edge_compare());

    std::size_t incorrect = 0;

    FOR_EACH( it, edges )
    {
        uint32_t set1 = sets.find_set(it->get<1>()); // region A
        uint32_t set2 = sets.find_set(it->get<2>()); // region B

        if ( set1 != set2 )
        {
            std::size_t n_pair_same = 0;
            std::size_t n_pair_diff = sizes[set1]*sizes[set2];

            FOR_EACH( sit, contains[set1] )
            {
                if ( contains[set2].find(sit->first) != contains[set2].end() )
                {
                    std::size_t pairs = sit->second * contains[set2][sit->first];
                    n_pair_diff -= pairs;
                    if ( sit->first != 0 )
                    {
                        n_pair_same += pairs;
                    }
                }
            }

            if ( (it->get<0>()) > 0.5 )
            {
                incorrect += n_pair_diff;
            }
            else
            {
                incorrect += n_pair_same;
            }

            double* p = it->get<3>();

            // positive (y = 1), splitter
            double dl = -std::max(0.0,0.5+margin-(it->get<0>()));
            *p   += dl*n_pair_same;
            loss += dl*dl*0.5*n_pair_same;

            // negative (y = 0), merger
            dl = std::max(0.0,(it->get<0>())-0.5+margin);
            *p   += dl*n_pair_diff;
            loss += dl*dl*0.5*n_pair_diff;

            uint32_t new_set = sets.join(set1, set2);
            sizes[set1] += sizes[set2];
            sizes[set2] = 0;

            FOR_EACH( sit, contains[set2] )
            {
                contains[set1][sit->first] += sit->second;
            }
            contains[set2].clear();

            std::swap(sizes[new_set], sizes[set1]);
            std::swap(contains[new_set], contains[set1]);
        }
    }

    return std::make_pair(affs, loss/n_pairs);
}


}} // namespace zi::znn

#endif // ZNN_MALIS_HPP_INCLUDED
