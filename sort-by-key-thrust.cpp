#include <iostream>
#include <vector>
#include <random>

#define VEXCL_BACKEND_CUDA
#include <vexcl/vexcl.hpp>

template <typename Key, typename Val>
void thrust_sort_by_key(Key *key_begin, Key *key_end, Val *val_begin);

template <typename Key, typename Val>
std::pair<Key*, Val*> thrust_reduce_by_key(
        const Key *key_begin, const Key *key_end, const Val *val_begin,
        Key *key_output, Val *val_output
        );

int main() {
    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    const size_t n = 1 << 24;

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

    int    *k_begin = k(0).raw_ptr();
    int    *p_begin = p(0).raw_ptr();
    double *v_begin = v(0).raw_ptr();
    double *s_begin = s(0).raw_ptr();

    int    *p_end;
    double *s_end;

    prof.tic_cl("GPU");
    thrust_sort_by_key(k_begin, k_begin + n, v_begin);
    std::tie(p_end, s_end) = thrust_reduce_by_key(k_begin, k_begin + n, v_begin, p_begin, s_begin);
    prof.toc("GPU");

    size_t n_keys = p_end - p_begin;

    prof.tic_cl("Transfer");
    vex::copy(p.begin(), p.begin() + n_keys, key.begin());
    vex::copy(s.begin(), s.begin() + n_keys, val.begin());
    prof.toc("Transfer");

    double delta = 0;
    for(size_t i = 0; i < n_keys; ++i)
        delta += fabs(val[i] - sum[key[i]]);

    std::cout
        << "delta = " << delta << std::endl
        << prof << std::endl;
}