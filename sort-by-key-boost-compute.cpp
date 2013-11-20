#include <iostream>
#include <vector>
#include <random>
#include <cstdlib>

#include <vexcl/vexcl.hpp>
#include <boost/compute.hpp>

int main(int argc, char *argv[]) {
    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    const size_t n = (argc > 1) ? atoi(argv[1]) : (1 << 10);

    std::default_random_engine             rng(0);
    std::uniform_int_distribution<int>     irnd(0, 99);
    std::uniform_real_distribution<double> drnd(0.0, 1.0);

    std::vector<int>    key(n);
    std::vector<double> val(n);
    std::vector<double> sum(100, 0.0);

    std::generate(key.begin(), key.end(), [&]() { return irnd(rng); });
    std::generate(val.begin(), val.end(), [&]() { return drnd(rng); });

    vex::profiler<> prof(ctx);

    prof.tic_cpu("CPU");
    for(size_t i = 0; i < n; ++i) sum[key[i]] += val[i];
    prof.toc("CPU");

    prof.tic_cl("Transfer");
    vex::vector<int>    k(ctx, key), p(ctx, 100);
    vex::vector<double> v(ctx, val), s(ctx, 100);
    prof.toc("Transfer");

    boost::compute::command_queue bcq( ctx.queue(0)() );
    boost::compute::buffer        bck( k(0).raw() );
    boost::compute::buffer        bcv( v(0).raw() );
    boost::compute::buffer        bcp( p(0).raw() );
    boost::compute::buffer        bcs( s(0).raw() );

    prof.tic_cl("GPU");
    boost::compute::sort_by_key(
            boost::compute::make_buffer_iterator<int>   (bck, 0),
            boost::compute::make_buffer_iterator<int>   (bck, n),
            boost::compute::make_buffer_iterator<double>(bcv, 0),
            bcq
            );
    // std::tie(p_end, s_end) = thrust_reduce_by_key(k_begin, k_begin + n, v_begin, p_begin, s_begin);
    prof.toc("GPU");

    size_t n_keys = 100;//p_end - p_begin;

    prof.tic_cl("Transfer");
    vex::copy(p.begin(), p.begin() + n_keys, key.begin());
    vex::copy(s.begin(), s.begin() + n_keys, val.begin());
    prof.toc("Transfer");

    double delta = 0;
    /*
    for(size_t i = 0; i < n_keys; ++i)
        delta += fabs(val[i] - sum[key[i]]);
    */

    std::cout << k << std::endl;
    std::cout
        << "delta = " << delta << std::endl
        << prof << std::endl;
}
