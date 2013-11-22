#include <iostream>
#include <vector>
#include <random>

#include <eigen3/Eigen/SparseCore>

#define VEXCL_BACKEND_CUDA
#include <vexcl/vexcl.hpp>

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

    // These will hold the reduced results.
    // NB: we do not know how much space we need. It would be easier to guess
    // with actual regular grid.
    vex::vector<int>    r(ctx, n);
    vex::vector<int>    c(ctx, n);
    vex::vector<double> v(ctx, n);
    prof.toc("Transfer");

    prof.tic_cl("Thrust");
    // Sort key-value pairs on the GPU:
    thrust_sort_by_key(
            full_r(0).raw_ptr(), full_r(0).raw_ptr() + n,
            full_c(0).raw_ptr(),
            full_v(0).raw_ptr()
            );

    // Reduce values on the GPU:
    double *v_end = thrust_reduce_by_key(
            full_r(0).raw_ptr(), full_r(0).raw_ptr() + n,
            full_c(0).raw_ptr(),
            full_v(0).raw_ptr(),
            r(0).raw_ptr(), c(0).raw_ptr(), v(0).raw_ptr()
            );
    prof.toc("Thrust");

    // Check that we got correct results:
    std::cout
        << "Host nnz = " << A.nonZeros()
        << "; GPU nnz = " << (v_end - v(0).raw_ptr())
        << std::endl;

    double *host_val = A.valuePtr();
    double delta = 0;
    for(size_t i = 0; i < 1024; ++i) {
        int j = irnd(rng);
        delta += fabs(v[j] - host_val[j]);
    }

    std::cout
        << "delta = " << delta << std::endl
        << prof << std::endl;
}
