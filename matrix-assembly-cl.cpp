#include <iostream>
#include <vector>
#include <random>

#include <eigen3/Eigen/SparseCore>
#include <vexcl/vexcl.hpp>

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    const size_t n = (argc > 1) ? atoi(argv[1]) : (1 << 20); // # of non zeros
    const size_t m = (argc > 2) ? atoi(argv[2]) : n / 4;     // # of grid nodes

    // Generate matrix in COO format on host.
    std::default_random_engine             rng(0);
    std::uniform_int_distribution<int>     irnd(0, m - 1);
    std::uniform_real_distribution<double> drnd(0.0, 1.0);

    std::vector<int>    row(n);
    std::vector<int>    col(n);
    std::vector<double> val(n);

    std::generate(row.begin(), row.end(), [&]() { return irnd(rng); });
    std::generate(col.begin(), col.end(), [&]() { return irnd(rng); });
    std::generate(val.begin(), val.end(), [&]() { return drnd(rng); });

    vex::profiler<> prof(ctx);

    // Reduce values on host:
    prof.tic_cpu("CPU");
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(m, m);
    A.reserve(Eigen::VectorXi::Constant(m, 2 * n / m));

    for(size_t i = 0; i < val.size(); ++i)
        A.coeffRef(row[i], col[i]) += val[i];

    A.makeCompressed();
    prof.toc("CPU");

    prof.tic_cl("Transfer");
    // Send the key-value pairs to the GPU.
    vex::vector<int>    full_r(ctx, row);
    vex::vector<int>    full_c(ctx, col);
    vex::vector<double> full_v(ctx, val);

    // Get matrix shape from host.
    vex::vector<int>    r(ctx, m + 1,        A.outerIndexPtr());
    vex::vector<int>    c(ctx, A.nonZeros(), A.innerIndexPtr());
    prof.toc("Transfer");

    prof.tic_cl("GPU");
    // Do the assembling.
    vex::vector<double> v(ctx, A.nonZeros());

    v = 0;

    vex::backend::kernel( ctx.queue(0),
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
            "kernel void assemble(\n"
            "  ulong n,\n"
            "  global const int    * I,\n"
            "  global const int    * J,\n"
            "  global const double * V,\n"
            "  global const int    * R,\n"
            "  global const int    * C,\n"
            "  global       double * val\n"
            ")\n"
            "{\n"
            "  for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n"
            "    int    i = I[idx];\n"
            "    int    j = J[idx];\n"
            "    double v = V[idx];\n"
            "    for(int k = R[i], row_end = R[i+1]; k < row_end; ++k) {\n"
            "      if(C[k] == j) {\n"
            "        AtomicAdd(val + k, v);\n"
            "        break;\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n",
            "assemble"
            )(ctx.queue(0), n, full_r(0), full_c(0), full_v(0), r(0), c(0), v(0));

    prof.toc("GPU");

    // Check that we got correct results:
    std::cout << "nnz = " << A.nonZeros() << std::endl;

    double delta = 0;
    std::uniform_int_distribution<int> crnd(0, A.nonZeros() - 1);
    for(size_t i = 0; i < 1024; ++i) {
        int    k = crnd(rng);
        delta += fabs(v[k] - A.valuePtr()[k]);
    }

    std::cout
        << "delta = " << delta << std::endl
        << prof << std::endl;
}
