#include <iostream>
#include <vector>
#include <random>

#include <vexcl/vexcl.hpp>

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

    // This will hold the reduced results:
    vex::vector<double> s(ctx, m);
    prof.toc("Transfer");

    s = 0;

    // Compile and launch reduction kernel.
    prof.tic_cl("GPU (Atomic)");
    vex::backend::kernel krn( ctx.queue(0),
            "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
            "#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable\n"
            "void AtomicAdd(__global double *val, double delta) {\n"
            "  union {\n"
            "    double f;\n"
            "    ulong  i;\n"
            "  } old;\n"
            "  union {\n"
            "    double f;\n"
            "    ulong  i;\n"
            "  } new;\n"
            "  do {\n"
            "    old.f = *val;\n"
            "    new.f = old.f + delta;\n"
            "  } while (atom_cmpxchg ( (volatile __global ulong *)val, old.i, new.i) != old.i);\n"
            "}\n"
            "kernel void atomic_reduce(\n"
            "  ulong n,\n"
            "  global const int    * key,\n"
            "  global const double * val,\n"
            "  global double * sum\n"
            ")\n"
            "{\n"
            "  for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0))\n"
            "    AtomicAdd(sum + key[idx], val[idx]);\n"
            "}\n",
            "atomic_reduce"
            );
    krn.push_arg(n);
    krn.push_arg(k(0));
    krn.push_arg(v(0));
    krn.push_arg(s(0));

    krn(ctx.queue(0));
    prof.toc("GPU (Atomic)");

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
