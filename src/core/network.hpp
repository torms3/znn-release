//
// Copyright (C) 2014  Aleksandar Zlateski <zlateski@mit.edu>
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

#ifndef ZNN_NETWORK_HPP_INCLUDED
#define ZNN_NETWORK_HPP_INCLUDED

#include "utils.hpp"
#include "node.hpp"
#include "../front_end/data_provider/data_providers.hpp"
#include "../cost_fn/cost_fns.hpp"
#include "../front_end/learning_monitor/learning_monitor.hpp"
#include "../front_end/net_builder.hpp"
#include "../front_end/forward_scanner/forward_scanners.hpp"
#include "../front_end/options.hpp"

#include <zi/concurrency.hpp>
#include <zi/utility/assert.hpp>
#include <zi/time.hpp>

#include <boost/lexical_cast.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>

namespace zi {
namespace znn {

class network
{
private:    
    net_ptr net_;
    
    zi::task_manager::prioritized tm_;

    std::size_t lastn_;

    // input data
    std::map<int,data_provider_ptr> inputs_;

    // forward scanner
    std::map<int,forward_scanner_ptr> scanners_;

    // option parser    
    options_ptr op;    

    // iteration number
    std::size_t n_iter_;
    
    // cost function
    cost_fn_ptr cost_fn_;

    // monitoring
    learning_monitor_ptr    train_monitor_;
    learning_monitor_ptr    test_monitor_ ;

    // learning rate parameter
    double eta_;


private:
    void load_inputs()
    {
	batch_list batches = op->get_batch_range();
        FOR_EACH( it, batches )
        {
            load_input(*it);
        }

        if ( inputs_.empty() )
        {
            std::string what = "No training data";
            throw std::logic_error(what);
        }
    }

    void load_input( int n )
    {
        // data spec
        std::string batch_name = op->data_path;
        std::size_t batch_num  = 0;

        if ( op->batch_template )
        {
            batch_num = n;
        }
        else
        {
            batch_name += boost::lexical_cast<std::string>(n);
        }

        batch_name += ".spec";

        // data spec parser        
        data_spec_parser parser(batch_name,batch_num);

        // loading
        std::cout << "[network] load_input" << std::endl;
        std::cout << "Loading [" << batch_name << "]" << std::endl;
        
        // inputs
        std::vector<vec3i> in_szs = net_->input_sizes();
        std::cout << "Input sizes:  ";
        FOR_EACH( it, in_szs )
        {
             std::cout << (*it) << " ";
        }
        std::cout << std::endl;

        // outputs
        std::vector<vec3i> out_szs = net_->output_sizes();
        std::cout << "Output sizes: ";
        FOR_EACH( it, out_szs )
        {
             std::cout << (*it) << " ";
        }
        std::cout << "\n\n";

        if ( op->dp_type == "volume" )
        {
            volume_data_provider* dp =
                new volume_data_provider(parser,in_szs,out_szs,op->mirroring);
            
            dp->data_augmentation(op->data_aug);
            
            inputs_[n] = data_provider_ptr(dp);
        }
        else if ( op->dp_type == "affinity" )
        {
            affinity_data_provider* dp =
                new affinity_data_provider(parser,in_szs,out_szs,op->mirroring);
            
            dp->data_augmentation(op->data_aug);
            
            inputs_[n] = data_provider_ptr(dp);   
        }
        else
        {
            std::string what = "Unknown data provider type [" + op->dp_type + "]";
            throw std::invalid_argument(what);
        }
    }

    void load_test_inputs()
    {        
        FOR_EACH( it, op->test_range )
        {
            int n = *it;
            std::cout << "batch" << n << std::endl;
            load_test_input(n);            
            std::cout << "batch" << n << " loaded." << std::endl;
        }

        if ( scanners_.empty() )
        {
            std::string what = "No testing data";
            throw std::logic_error(what);
        }
    }

    void load_test_input( int n )
    {
        // data spec
        std::string batch_name = op->data_path;
        std::size_t batch_num  = 0;

        if ( op->batch_template )
        {
            batch_num = n;
        }
        else
        {
            batch_name += boost::lexical_cast<std::string>(n);
        }
        
        batch_name += ".spec";

        // data spec parser        
        data_spec_parser parser(batch_name,batch_num);
        
        // inputs
        std::vector<vec3i> in_szs = net_->input_sizes();
        std::cout << "Input sizes: ";
        FOR_EACH( it, in_szs )
        {
             std::cout << (*it) << " ";
        }
        std::cout << std::endl;

        // outputs
        std::vector<vec3i> out_szs = net_->output_sizes();
        std::cout << "Output sizes: ";
        FOR_EACH( it, out_szs )
        {
             std::cout << (*it) << " ";
        }
        std::cout << std::endl;
        
        if ( op->scanner == "volume" )
        {
            scanners_[n] = forward_scanner_ptr(new 
                volume_forward_scanner(parser,
                                       in_szs,out_szs,
                                       op->scan_offset,
                                       op->subvol_dim,
                                       op->mirroring));

            if ( op->scan_fmaps )
                scanners_[n]->set_feature_map_scanner(net_,op->scan_all);
        }
        else
        {
            std::string what = "Unknown forward scanner type [" + op->scanner + "]";
            throw std::invalid_argument(what);
        }
    }    


// input sampling
private:
    sample_ptr random_sample( batch_list lst )
    {
        int e = lst[rand() % lst.size()];
        ZI_ASSERT(inputs_.find(e) != inputs_.end());
        return inputs_[e]->random_sample();
    }


private:
    double run_n_times(std::size_t n, sample_ptr s, bool scanning)
    {
        zi::wall_timer wt;

        for ( std::size_t i = 0; i < n; ++i )
        {
            // forward pass
            std::list<double3d_ptr> v = run_forward(s->inputs);

            if ( !scanning )
            {
                // backward pass
                run_backward(s);
            }
        }

        return wt.elapsed<double>();
    }

    void force_fft(bool b = true)
    {
        net_->force_fft(b);
    }
    
    // enable the optimization for scanning
    void optimize_for_training(bool scanning = false)
    {
        // [kisuklee] temporary
        #define FILE_AND_CONSOLE( ofs, oss )        \
            {                                       \
                      ofs << oss.str() << "\n";     \
                std::cout << oss.str() << "\n";     \
                oss.clear(); oss.str("");           \
            }

        std::string fname = op->save_path + "optimization.profile";
        std::ofstream fout(fname.c_str(), std::ios::out);

        // force fft
        force_fft(true);
        
        // warming up
        sample_ptr sample;
        if ( scanning )
        {
            sample = random_sample(op->train_range);
        }
        else
        {
            sample = random_sample(op->test_range);
        }

        std::ostringstream line;
        line << "Warmup ffts (make fftplans): " 
             << run_n_times(1, sample, scanning);
        FILE_AND_CONSOLE( fout, line );

        double approx = run_n_times(1, sample, scanning);
        std::size_t run_times = 
            static_cast<std::size_t>(static_cast<double>(5)/approx);
        if ( run_times < 2 )
        {
            run_times = 2;
        }

        line << "Will run " << run_times << " iterations per test.";
        FILE_AND_CONSOLE( fout, line );

        double best = run_n_times(run_times, sample, scanning);
        line << "Best so far (all ffts): " << best;
        FILE_AND_CONSOLE( fout, line );        
        
        // temporary solution for layer-wise fft opimization
        // should be refactored
        FOR_EACH( it, net_->node_groups_ )
        {
            if ( (*it)->is_crop() ) continue;

            if ( (*it)->count_in_connections() > 0 )
            {
                line << "Testing node_group [" << (*it)->name() << "] ...";
                FILE_AND_CONSOLE( fout, line );

                (*it)->receives_fft(false);
                
                double t = run_n_times(run_times, sample, scanning);
                line << "   when using ffts   : " << best << "\n"
                     << "   when using bf_conv: " << t;                

                if ( t < best )
                {
                    best = t;
                    line << "   will use bf_conv";                    
                }
                else
                {
                    line << "   will use ffts";                    
                    (*it)->receives_fft(true);
                }
                FILE_AND_CONSOLE( fout, line );
            }   
        }

        line << "Optimization done.";
        FILE_AND_CONSOLE( fout, line );

        fout.close();
    }


// training parameters
private:    
    void force_eta( double eta )
    {
        std::cout << "Force learning rate parameter to be " << eta << std::endl;
        
        net_->set_learning_rate(eta);
    }

    void set_momentum( double mom )
    {
        std::cout << "Momentum: " << mom << std::endl;
        
        if ( mom > static_cast<double>(0) )
        {
            net_->set_momentum(mom);
        }
    }

    void set_weight_decay( double wc )
    {
        std::cout << "Weight decay: " << wc << std::endl;
        
        net_->set_weight_decay(wc);
    }


public:
    void prepare_training()
    {
        // resume training
        {
            n_iter_ = train_monitor_->load_state(op->load_path);
            n_iter_ = n_iter_ + 1;

            test_monitor_->load_state(op->load_path);
        }

        // learning rate parameter
        eta_ = op->force_eta;
        if ( eta_ > 0 ) force_eta(eta_);

        // momentum setting
        set_momentum(op->momentum);

        // weight decay setting
        set_weight_decay(op->wc_factor);
        
        // This must precede the setup_fft() routine
        // set minibatch size
        if ( op->minibatch )
        {
            set_minibatch_size(op->outsz);
        }

        // optimize for training (fft vs non fft)
        setup_fft();

        // time-stamped network weight
        if ( op->weight_idx > 0 )
        {
            boost::filesystem::path hist_dir(op->hist_path);
            boost::filesystem::path load_dir(op->load_path);

            net_->reload_weight(op->weight_idx-1);
            if ( !op->hist_path.empty() )
            {
                net_->save(op->hist_path);
            }

            // train information
            export_train_information();
        }
    }

    void set_minibatch_size( vec3i sz )
    {
        set_minibatch_size(sz[0]*sz[1]*sz[2]);
    }

    void set_minibatch_size( double B )
    {        
        net_->set_minibatch_size(B);
    }

    void setup_fft()
    {
        if ( op->force_fft )
        {
            force_fft(true);
        }
        else
        {
            if ( op->optimize_fft )
            {
                optimize_for_training();

                // Since optimization screws up the weights, 
                // re-initializes the weights again.
                net_->initialize_weight();
                net_->initialize_momentum();
                net_->save(op->save_path);
            }
        }        
    }


public:
    void train()
    {
        // save network spec
        net_->save(op->save_path);

        // load data batches for training
        load_inputs();
        
        // learning rate, momentum, weight decay, fft ...
        prepare_training();

        std::cout << "Iterations: " << op->n_iters << std::endl;

        zi::wall_timer wt;
        for ( std::size_t tick = 1; n_iter_ <= op->n_iters; ++tick, ++n_iter_ )
        {
            sample_ptr s = random_sample(op->train_range);

            // forward pass
            std::list<double3d_ptr> v = run_forward(s->inputs);
           
            // update training monitor
            train_monitor_->update(v, s->labels, s->masks, op->cls_thresh);

            // backward pass
            run_backward(s);

            // updates/sec
            if ( n_iter_ % op->check_freq == 0 )
            {
                double speed = wt.elapsed<double>()/tick;
                std::cout << "[Iter: " << std::setw(count_digit(op->n_iters)) 
                          << n_iter_ << "] " << speed << " secs/update\t";
                train_monitor_->push_speed(speed);
            }

            // check error
            if ( n_iter_ % op->check_freq == 0 )
            {
                // push, report & save training info
                train_monitor_->check(op->save_path,n_iter_);

                net_->save(op->save_path);

                wt.restart();
                tick = 0;
            }

            // test
            if ( n_iter_ % op->test_freq == 0 )
            {
                test_check();                
                net_->save(op->save_path,true);
                wt.restart();
                tick = 0;
            }

            // learning rate annealing
            if ( (op->anneal_freq > 0) && (n_iter_ % op->anneal_freq == 0) )
            {
                eta_ = op->anneal_factor*eta_;
                force_eta(eta_);
            }
        }
    }
   
    void forward_scan()
    {
        if ( op->time_series != vec3i::zero )
        {
            time_series_scan();
            return;
        }

        // load inputs for forward scanning
        load_test_inputs();

        prepare_testing();

        FOR_EACH( it, op->test_range )
        {
            int idx = *it;
            forward_scanner_ptr scanner = scanners_[idx];

            std::cout << "\n[Batch " << idx << "]\n\n";

            std::list<double3d_ptr> inputs;
            std::list<double3d_ptr> outputs;

            std::size_t i = 0;
            while ( scanner->pull(inputs) )
            {
                std::cout << "Subvolume " << ++i << std::endl;
                std::cout << "Input sizes: ";
                FOR_EACH( jt, inputs )
                {
                     std::cout << size_of(*jt) << " ";
                }
                std::cout << std::endl;

                zi::wall_timer wt;
                outputs = run_forward(inputs);

                scanner->push(outputs);
                std::cout << "Elapsed time: " << wt.elapsed<double>()
                          << " secs" << std::endl;
            }

            // output file name
            std::ostringstream batch;
            batch << op->outname << idx << op->subname;

            // save feature maps
            if ( op->scan_fmaps )
            {
                const std::string& fmaps_path = op->save_path + "fmaps/";
                boost::filesystem::path fmaps_dir(fmaps_path);
                if ( !boost::filesystem::exists(fmaps_dir) )
                {
                    boost::filesystem::create_directory(fmaps_dir);
                }
                scanner->save_feature_maps(fmaps_path + batch.str() + ".",op->scan_all);
            }
            else // save output
            {
                scanner->save(op->save_path + batch.str());
            }
        }
    }

    void test_check()
    {
        if( op->test_range.empty() )
            return;

        // dropout
        net_->set_inference(true);

        // test loop
        for ( std::size_t i = 1; i <= op->test_samples; ++i )
        {
            sample_ptr s = random_sample(op->test_range);

            // forward pass
            std::list<double3d_ptr> v = run_forward(s->inputs);
            
            // error computation
            test_monitor_->update(v, s->labels, s->masks);
        }

        // dropout
        net_->set_inference(false);

        std::cout << "<<<<<<<<<<<<< TEST >>>>>>>>>>>>>" << std::endl;
        test_monitor_->check(op->save_path, n_iter_);
        std::cout << "<<<<<<<<<<<<< TEST >>>>>>>>>>>>>" << std::endl;
    }


private:
    void prepare_testing()
    {
        // optimize for training (fft vs non fft)
        setup_fft();

        // Force load for forward scanning
        if ( op->force_load )
        {
            net_->force_load();
        }

        // DropOut
        net_->set_inference(true);

        // time-stamped network weight
        if ( op->weight_idx > 0 )
        {
            boost::filesystem::path hist_dir(op->hist_path);
            boost::filesystem::path load_dir(op->load_path);
            STRONG_ASSERT(!boost::filesystem::equivalent(hist_dir,load_dir));

            net_->reload_weight(op->weight_idx-1);
            if ( !op->hist_path.empty() )
            {
                net_->save(op->hist_path);    
            }

            // train information
            export_train_information();
        }
    }

    void export_train_information()
    {
        STRONG_ASSERT(op->test_freq * op->check_freq > 0);
                
        std::size_t test_idx  = op->weight_idx;
        std::size_t train_idx = op->weight_idx * op->test_freq / op->check_freq;

        std::cout << "train index: " << train_idx << std::endl;
        train_monitor_->load_state(op->load_path,train_idx);
        train_monitor_->save_state(op->hist_path);
        
        std::cout << "test  index: " << test_idx << std::endl;
        test_monitor_->load_state(op->load_path,test_idx);
        test_monitor_->save_state(op->hist_path);
    }

    void time_series_scan()
    {
        load_test_inputs();
        
        setup_fft();
        
        if ( op->force_load )
        {
            net_->force_load();
        }

        std::size_t head    = op->time_series[0];
        std::size_t step    = op->time_series[1];
        std::size_t tail    = op->time_series[2];
        std::size_t counter = 1;
        for ( std::size_t i = head; i <= tail; i += step, ++counter )
        {
            std::cout << "\n[Time series index = " << i << "] ";

            // reload time-stamped network weight
            net_->reload_weight(i-1);

            // forward scanning
            FOR_EACH( it, op->test_range )
            {
                int idx = *it;
                forward_scanner_ptr scanner = scanners_[idx];

                std::list<double3d_ptr> inputs;
                std::list<double3d_ptr> outputs;

                while ( scanner->pull(inputs) )
                {
                    outputs = run_forward(inputs);
                    scanner->push(outputs);
                }
                
                std::ostringstream batch;
                batch << op->save_path << op->outname << idx << op->subname;
                scanner->save(batch.str(),counter);
            }            
        }
    }


public:
    network( options_ptr _op )
        : net_(new net)        
        , tm_(_op->n_threads)
        , lastn_(0)
        , inputs_()
        , op(_op)
        , n_iter_(1)
        , cost_fn_(_op->create_cost_function())
        , train_monitor_(new learning_monitor("train",_op->create_cost_function()))
        , test_monitor_(new learning_monitor("test",_op->create_cost_function()))
        , eta_(0.01)        
    {
        if ( !construct_network() )
        {
            std::string what = "Failed to construct net";
            throw std::invalid_argument(what);
        }

        tm_.start();

        // time seed for rand()
        srand(time(NULL));
    }

    ~network()
    {
        tm_.join();
    }

    // will run the forward pass and will return the computed guess
    std::list<double3d_ptr> run_forward(std::list<double3d_ptr> inputs)
    {
        std::list<double3d_ptr>::iterator iit = inputs.begin();
        FOR_EACH( it, net_->inputs_ )
        {
            lastn_ = (*it)->run_forward(*iit++, &tm_);
        }

        // this is where we wait for the last thread to finish
        // that's the output node
        std::list<double3d_ptr> ret;
        FOR_EACH( it, net_->outputs_ )
        {
            ret.push_back((*it)->wait(lastn_));
        }
        ret = net_->get_outputs(lastn_,op->softmax);
        return ret;
    }
    
    void run_backward( sample_ptr s )
    {
        std::list<double3d_ptr> outs = net_->get_outputs(lastn_,op->softmax);

        // compute gradient using a given cost function
        std::list<double3d_ptr> grads = cost_fn_->gradient(outs,s->labels,s->masks);
        
        // rebalancing (global)
        if ( op->rebalance )
        {
            std::list<double3d_ptr>::iterator wit = s->wmasks.begin();
            FOR_EACH( it, grads )
            {
                volume_utils::elementwise_mul_by(*it, *wit++);
            }
        }
        else if ( op->patch_bal ) // patch-wise rebalancing
        {
            // default cross-entropy is multinomial (1-of-K coding)
            if ( op->cost_fn == "cross_entropy" )
            {
                // TODO: Should exclude the locations that are masked out
                double3d_ptr rbmask = 
                    volume_utils::multinomial_rebalance_mask(s->labels);

                FOR_EACH( it, grads )
                {
                    volume_utils::elementwise_mul_by(*it,rbmask);
                }
            }
            else
            {
                // TODO: Should exclude the locations that are masked out
                std::list<double3d_ptr> rbmask = 
                    volume_utils::binomial_rebalance_mask(s->labels);

                std::list<double3d_ptr>::iterator rbmit = rbmask.begin();
                FOR_EACH( it, grads )
                {
                    volume_utils::elementwise_mul_by(*it,*rbmit++);
                }
            }
        }

        if ( op->norm_grad )
        {
            FOR_EACH( it, grads )
            {
                volume_utils::normalize(*it);
            }
        }        

        std::list<double3d_ptr>::iterator git = grads.begin();
        FOR_EACH( it, net_->outputs_ )
        {            
            lastn_ = (*it)->run_backward(*git++,&tm_);
        }

        // this is where we wait for the last thread to finish
        // that's the input node
        FOR_EACH( it, net_->inputs_ )
        {
            (*it)->wait(lastn_);
        }
    }


private:
    // net construction
    bool construct_network()
    {
        // net builder
        net_builder builder(op->config_path);

        // build network
        if ( builder.operable() )
        {
            net_ = builder.build(op->load_path);
            if ( !op->out_filter ) net_->disable_output_filtering();
            net_->initialize(op->outsz);
            
            // save net architecture 
            std::string fname = op->save_path + "architecture.txt";
            std::ofstream fout(fname.c_str(), std::ios::out);
            net_->display(fout);
            fout.close();
        }

        return net_->initialized();
    }


public:    
    // test loop: test whatever you want
    void test_loop()
    {   
    }

}; // class network

}} // namespace zi::znn

#endif // ZNN_NETWORK_HPP_INCLUDED
