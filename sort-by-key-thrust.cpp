#include <iostream>
#include <vector>
#include <random>

#define VEXCL_BACKEND_CUDA
#include <vexcl/vexcl.hpp>

//---------------------------------------------------------------------------
// The following templates are defined and instantiated in
// sort-by-key-thrust.cu
//---------------------------------------------------------------------------
template <typename Key, typename Val>
void thrust_sort_by_key(Key *key_begin, Key *key_end, Val *val_begin);

template <typename Key, typename Val>
std::pair<Key*, Val*> thrust_reduce_by_key(
        const Key *key_begin, const Key *key_end, const Val *val_begin,
        Key *key_output, Val *val_output
        );

//---------------------------------------------------------------------------
int main() {
    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    const size_t n = 1 << 24;
    const size_t m = 100;

    // Generate key-value pairs on host. Keys are integers in [0,99] range.
    std::default_random_engine             rng(0);
    std::uniform_int_distribution<int>     irnd(0, m - 1);
    std::uniform_real_distribution<double> drnd(0.0, 1.0);

    std::vector<int>    key(n);
    std::vector<double> val(n);
    std::vector<double> sum(m, 0.0);

    std::generate(key.begin(), key.end(), [&]() { return irnd(rng); });
    std::generate(val.begin(), val.end(), [&]() { return drnd(rng); });

    vex::profiler<> prof(ctx);

    // Reduce values on host:
    prof.tic_cpu("CPU");
    for(size_t i = 0; i < n; ++i) sum[key[i]] += val[i];
    prof.toc("CPU");

    prof.tic_cl("Transfer");
    // Send the key-value pairs to the GPU.
    vex::vector<int>    k(ctx, key);
    vex::vector<double> v(ctx, val);

    // These will hold the reduced results:
    vex::vector<int>    p(ctx, m);
    vex::vector<double> t(ctx, m);
    vex::vector<double> s(ctx, m);
    prof.toc("Transfer");

    // Extract raw pointers for thrust.
    int    *k_begin = k(0).raw_ptr();
    int    *p_begin = p(0).raw_ptr();
    double *v_begin = v(0).raw_ptr();
    double *t_begin = t(0).raw_ptr();

    int    *p_end;
    double *t_end;

    prof.tic_cl("Thrust");
    // Sort key-value pairs on the GPU:
    thrust_sort_by_key(k_begin, k_begin + n, v_begin);

    // Reduce values on the GPU:
    std::tie(p_end, t_end) = thrust_reduce_by_key(
            k_begin, k_begin + n, v_begin, p_begin, t_begin);

    s = 0;
    vex::permutation(
            vex::vector<int>(ctx.queue(0), p(0), p_end - p_begin)
            )(s) = t;
    prof.toc("Thrust");

    prof.tic_cl("Transfer");
    vex::copy(s.begin(), s.end(), val.begin());
    prof.toc("Transfer");

    // Check that we got correct results:
    double delta = 0;
    for(size_t i = 0; i < m; ++i)
        delta += fabs(val[i] - sum[i]);

    std::cout
        << "delta = " << delta << std::endl
        << prof << std::endl;
}
