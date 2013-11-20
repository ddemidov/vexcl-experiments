#include <iostream>
#include <vector>
#include <random>

#include <eigen3/Eigen/SparseCore>

#include <vexcl/vexcl.hpp>
#include <boost/compute.hpp>

//---------------------------------------------------------------------------
// The following templates are defined and instantiated in
// sort-by-key-thrust.cu
//---------------------------------------------------------------------------
template <typename Key1, typename Key2, typename Val>
void thrust_sort_by_key(
        Key1 *key1_begin, Key1 *key1_end, Key2 *key2_begin, Val *val_begin);

template <typename Key1, typename Key2, typename Val>
Val* thrust_reduce_by_key(
        const Key1 *key1_begin, const Key1 *key1_end, const Key2 *key2_begin,
        const Val *val_begin,
        Key1 *key1_output, Key2 *key2_output, Val *val_output
        );

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
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(n, n);
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

    prof.toc("Transfer");

    prof.tic_cl("GPU");
    // Get the matrix shape first.
    vex::vector<int> r(ctx, m + 1);
    r = 0;

    vex::backend::kernel nnz_per_row( ctx.queue(0),
            "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
            "kernel void nnz_per_row(\n"
            "  ulong n,\n"
            "  global const int * row,\n"
            "  global int * nnz\n"
            ")\n"
            "{\n"
            "  for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0))\n"
            "    atomic_inc(nnz + row[idx] + 1);\n"
            "}\n",
            "nnz_per_row"
            );

    nnz_per_row.push_arg(n);
    nnz_per_row.push_arg(full_r(0));
    nnz_per_row.push_arg(r(0));

    nnz_per_row( ctx.queue(0) );

    // Do a partial sum.
    {
        boost::compute::command_queue bcq( ctx.queue(0)() );
        boost::compute::buffer        bcr( r(0).raw() );
        boost::compute::partial_sum(
                boost::compute::make_buffer_iterator<int>(bcr, 0),
                boost::compute::make_buffer_iterator<int>(bcr, m + 1),
                boost::compute::make_buffer_iterator<int>(bcr, 0),
                bcq
                );
    }

    // Do the actual assembling.
    size_t nnz = r[m];

    vex::vector<int>    c(ctx, nnz);
    vex::vector<double> v(ctx, nnz);

    vex::tie(c, v) = std::make_tuple(-1, 0.0);

    vex::backend::kernel assemble( ctx.queue(0),
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
            "  global       int    * col,\n"
            "  global       double * val\n"
            ")\n"
            "{\n"
            "  for(size_t idx = get_global_id(0); idx < n; idx += get_global_size(0)) {\n"
            "    int    i = I[idx];\n"
            "    int    j = J[idx];\n"
            "    double v = V[idx];\n"
            "    int row_start = R[i];\n"
            "    int row_end   = R[i+1];\n"
            "    for(int k = row_start; k < row_end; ++k) {\n"
            "      int c = col[k];\n"
            "      if(c < 0) {\n"
            "        col[k] = j;\n"
            "        val[k] = v;\n"
            "        break;\n"
            "      } else if(c == j) {\n"
            "        AtomicAdd(val + k, v);\n"
            "        break;\n"
            "      }\n"
            "    }\n"
            "  }\n"
            "}\n",
            "assemble"
            );

    assemble.push_arg(n);
    assemble.push_arg(full_r(0));
    assemble.push_arg(full_c(0));
    assemble.push_arg(full_v(0));
    assemble.push_arg(r(0));
    assemble.push_arg(c(0));
    assemble.push_arg(v(0));
    prof.toc("GPU");

    // Check that we got correct results:
    std::cout
        << "Host nnz = " << A.nonZeros()
        << "; GPU nnz = " << nnz
        << std::endl;

    double *host_val = A.valuePtr();
    double delta = 0;
    for(size_t i = 0; i < 1024; ++i) {
        int    I = irnd(rng);
        int    p = r[I];
        int    J = c[p];
        double V = v[p];

        delta += fabs(V - A.coeffRef(I, J));
    }

    std::cout
        << "delta = " << delta << std::endl
        << prof << std::endl;
}
