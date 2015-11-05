//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_MALIS_2D_HPP_INCLUDED
#define ZNN_MALIS_2D_HPP_INCLUDED

#include "../core/types.hpp"
#include "../core/utils.hpp"
#include "../core/volume_utils.hpp"
#include "malis_def.hpp"
#include "utils/get_segmentation.hpp"

#include <boost/tuple/tuple.hpp>
#include <boost/tuple/tuple_comparison.hpp>
#include <zi/disjoint_sets/disjoint_sets.hpp>
#include <zi/utility/for_each.hpp>
#include <vector>
#include <functional>
#include <algorithm>

namespace zi {
namespace znn {

inline malis_return
malis_2d( std::list<double3d_ptr> true_affs,
          std::list<double3d_ptr> affs,
          double high = 0.99,
          double low = 0.3 )
{
    double loss = 0;
    long3d_ptr seg_ptr = get_segmentation(true_affs);
    long3d&    seg = *seg_ptr;

#if defined( DEBUG )
    std::cout << '\n';
    std::cout << "[Segmentation]" << std::endl;
    volume_utils::print_in_matlab_format(seg_ptr);
#endif

    ZI_ASSERT(affs.size()==2);

    vec3i s = volume_utils::volume_size(affs.front());

    // constrained to 2D
    ZI_ASSERT(s[2]==1);
    std::size_t n = s[0]*s[1];

    double3d_ptr xaff = affs.front();affs.pop_front();
    double3d_ptr yaff = affs.front();affs.pop_front();

#if defined( DEBUG )
    std::cout << "[size] = " << s << std::endl;
    std::cout << '\n';

    std::cout << "[x-affinity]" << std::endl;
    volume_utils::print_in_matlab_format(xaff);
    std::cout << "[y-affinity]" << std::endl;
    volume_utils::print_in_matlab_format(yaff);
#endif

    // merger weight
    std::list<double3d_ptr> mw;
    double3d_ptr xmw = volume_pool.get_double3d(s);
    double3d_ptr ymw = volume_pool.get_double3d(s);
    volume_utils::zero_out(xmw);mw.push_back(xmw);
    volume_utils::zero_out(ymw);mw.push_back(ymw);

    // splitter weight
    std::list<double3d_ptr> sw;
    double3d_ptr xsw = volume_pool.get_double3d(s);
    double3d_ptr ysw = volume_pool.get_double3d(s);
    volume_utils::zero_out(xsw);sw.push_back(xsw);
    volume_utils::zero_out(ysw);sw.push_back(ysw);

    // data structures for computing MALIS weight
    zi::disjoint_sets<uint32_t>     sets(n+1);
    std::vector<uint32_t>           sizes(n+1);
    std::map<uint32_t, uint32_t>    seg_sizes;
    std::vector<std::map<uint32_t, uint32_t> > contains(n+1);

    long3d_ptr ids_ptr = volume_pool.get_long3d(s);
    long3d& ids = *ids_ptr;

    std::size_t n_lbl_vert = 0;
    std::size_t n_pair_pos = 0;

    for ( std::size_t i = 0; i < n; ++i )
    {
        ids.data()[i] = i+1;
        sizes[i+1] = 1;
        contains[i+1][seg.data()[i]] = 1;

        // non-boundary
        if ( seg.data()[i] != 0 )
        {
            ++n_lbl_vert;
            ++seg_sizes[seg.data()[i]];

            // n*(n - 1)/2 = 1 + ... + n
            n_pair_pos += (seg_sizes[seg.data()[i]] - 1);
        }
    }

    // std::size_t n_pairs = n*(n-1)/2;
    std::size_t n_pair_lbl = n_lbl_vert*(n_lbl_vert-1)/2;
    // std::size_t n_pair_neg = n_pair_lbl - n_pair_pos;

    // (affinity value, vertex 1, vertex 2, merger weight, splitter weight)
    typedef boost::tuple<double,uint32_t,uint32_t,double*,double*> edge_type;
    typedef std::greater<edge_type> edge_compare;

    std::vector<edge_type> edges;
    edges.reserve(n*2); // 2D

    for ( std::size_t y = 0; y < s[1]; ++y )
        for ( std::size_t x = 0; x < s[0]; ++x )
        {
            if ( x > 0 )
            {
                double affinity = std::min((*xaff)[x][y][0],high);

                if ( affinity > low)
                {
                    edges.push_back(edge_type(affinity, ids[x-1][y][0],
                                          ids[x][y][0], &((*xmw)[x][y][0]),
                                          &((*xsw)[x][y][0])));
                }
            }

            if ( y > 0 )
            {
                double affinity = std::min((*yaff)[x][y][0],high);

                if ( affinity > low )
                {
                    edges.push_back(edge_type(affinity, ids[x][y-1][0],
                                          ids[x][y][0], &((*ymw)[x][y][0]),
                                          &((*ysw)[x][y][0])));
                }
            }
        }

    std::sort(edges.begin(), edges.end(), edge_compare());

#if defined( DEBUG )
    std::cout << "[# edges] = " << edges.size() << std::endl;
    std::cout << '\n';
    long3d_ptr ws_ptr = volume_pool.get_long3d(s[0],s[1],edges.size()+1);
    long3d&    ws = *ws_ptr;

    std::size_t cnt = 0;
    for ( std::size_t x = 0; x < s[0]; ++x )
        for ( std::size_t y = 0; y < s[1]; ++y )
            ws[x][y][0] = sets.find_set(ids[x][y][0]);
    ++cnt;
#endif

    // std::size_t incorrect = 0;
    std::size_t nTP = 0;
    std::size_t nFP = 0;
    std::size_t nFN = 0;
    std::size_t nTN = 0;

    // (B,N) or (B,B) pairs where B: boundary, N: non-boundary
    // std::size_t n_b_pairs = 0;

#if defined( DEBUG )
    long3d_ptr ts = volume_pool.get_long3d(s);
    volume_utils::zero_out(ts);
#endif
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
                // boundary
                if ( sit->first == 0 )
                {
                    std::size_t pairs = sit->second * sizes[set2];
                    n_pair_diff -= pairs;
                    // n_b_pairs   += pairs;
                }
                else // non-boundary
                {
                    if ( contains[set2].find(sit->first) != contains[set2].end() )
                    {
                        std::size_t pairs = sit->second * contains[set2][sit->first];
                        n_pair_diff -= pairs;
                        n_pair_same += pairs;
                    }

                    if ( contains[set2].find(0) != contains[set2].end() )
                    {
                        std::size_t pairs = sit->second * contains[set2][0];
                        n_pair_diff -= pairs;
                        // n_b_pairs   += pairs;
                    }
                }
            }

            if ( (it->get<0>()) > 0.5 )
            {
                // incorrect += n_pair_diff;
                nTP += n_pair_same;
                nFP += n_pair_diff;
            }
            else
            {
                // incorrect += n_pair_same;
                nTN += n_pair_diff;
                nFN += n_pair_same;
            }

            double* pm = it->get<3>(); // merger weight
            double* ps = it->get<4>(); // splitter weight

            *pm += n_pair_diff;
            *ps += n_pair_same;

            // bool hinge = true;
            // if ( hinge ) // hinge loss
            // {
            //     // delta(s_i,s_j) = 1
            //     double dl = std::max(0.0,0.5+margin-(it->get<0>()));
            //     *p   -= (dl > 0)*n_pair_same;
            //     loss += dl*n_pair_same;

            //     // delta(s_i,s_j) = 0
            //     dl = std::max(0.0,(it->get<0>())-0.5+margin);
            //     *p   += (dl > 0)*n_pair_diff;
            //     loss += dl*n_pair_diff;
            // }
            // else // square-square loss
            // {
            //     // delta(s_i,s_j) = 1
            //     double dl = -std::max(0.0,1.0-margin-(it->get<0>()));
            //     *p   += dl*n_pair_same;
            //     loss += dl*dl*0.5*n_pair_same;

            //     // delta(s_i,s_j) = 0
            //     dl = std::max(0.0,(it->get<0>())-margin);
            //     *p   += dl*n_pair_diff;
            //     loss += dl*dl*0.5*n_pair_diff;
            // }

            // normlize gradient
            // *p /= n_pair_lbl;

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

#if defined( DEBUG )
            for ( std::size_t x = 0; x < s[0]; ++x )
                for ( std::size_t y = 0; y < s[1]; ++y )
                    ws[x][y][cnt] = sets.find_set(ids[x][y][0]);

            ts->data()[it->get<2>()-1] = cnt;

            ++cnt;

            // std::cout << "[watershed " << cnt << "]" << std::endl;
            // long3d_ptr slice = volume_pool.get_long3d(s);
            // *slice = ws[boost::indices[range(0,s[0])][range(0,s[1])][range(cnt-1,cnt)]];
            // volume_utils::print_in_matlab_format(slice);
#endif
        }
    }

    // std::size_t n_eff_pairs = n_pairs - n_b_pairs;

    malis_metric metric;
    metric.loss = loss/n_pair_lbl;
    metric.nTP  = nTP;
    metric.nFP  = nFP;
    metric.nFN  = nFN;
    metric.nTN  = nTN;

    malis_weight malisw(mw,sw);
#if defined( DEBUG )
    malisw.ws_evolution = ws_ptr;
    malisw.time_step = ts;
#endif
    return std::make_pair(malisw, metric);
}


}} // namespace zi::znn

#endif // ZNN_MALIS_2D_HPP_INCLUDED
