#include <iostream>
#include <vector>
#include <random>

#include <boost/fusion/adapted/std_tuple.hpp>

#include <eigen3/Eigen/SparseCore>
#include <vexcl/vexcl.hpp>

//---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    const size_t n = (argc > 1) ? atoi(argv[1]) : (1 << 20); // # of non zeros
    const size_t m = (argc > 2) ? atoi(argv[2]) : n / 4;     // # of grid nodes
    const int    w = 4;

    // Generate matrix in COO format on host.
    std::default_random_engine             rng(0);
    std::uniform_int_distribution<int>     random_row(0, m - 1);
    std::uniform_int_distribution<int>     random_off(-w, w);
    std::uniform_real_distribution<double> random_val(0.0, 1.0);

    std::vector<int>    row(n);
    std::vector<int>    col(n);
    std::vector<double> val(n);

    for(size_t i = 0; i < n; ++i) {
        row[i] = random_row(rng);
        col[i] = row[i] + random_off(rng);
        val[i] = random_val(rng);
    }

    vex::profiler<> prof(ctx);

    // Reduce values on host:
    prof.tic_cpu("CPU");
    Eigen::SparseMatrix<double, Eigen::RowMajor> A(m, m);
    A.reserve(Eigen::VectorXi::Constant(m, 2 * w + 1));

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
    // Sort non-zeros by row/column:
    struct {
        typedef bool result_type; // Need this for g++-4.6 to work.

        VEX_FUNCTION(device, bool(int, int, int, int),
                "return (prm1 == prm3) ? (prm2 < prm4) : (prm1 < prm3);"
                );

        result_type operator()(int r1, int c1, int r2, int c2) const {
            return std::make_tuple(r1, c1) < std::make_tuple(r2, c2);
        }
    } less;

    vex::sort_by_key(std::tie(full_r, full_c), full_v, less);

    // Reduce non-zero values with same coordinates:
    vex::vector<int>    r;
    vex::vector<int>    c;
    vex::vector<double> v;

    VEX_FUNCTION(equal, bool(int, int, int, int), "return (prm1 == prm3) && (prm2 == prm4);");
    VEX_FUNCTION(plus,  double(double, double), "return prm1 + prm2;");

    size_t nnz = vex::reduce_by_key(std::tie(full_r, full_c), full_v, std::tie(r, c), v, equal, plus);

#if 0
    // Compress the row vector.
    vex::vector<int> ones(ctx, nnz);
    ones = 1;

    vex::vector<int> rx, nnz_by_row;
    vex::reduce_by_key(r, ones, rx, nnz_by_row);
    vex::exclusive_scan(nnz_by_row, nnz_by_row);
#endif
    prof.toc("GPU");

    // Check that we got correct results:
    std::cout << "nnz = " << A.nonZeros() << "/" << nnz << std::endl;

    double delta = 0;
    std::uniform_int_distribution<int> random_idx(0, A.nonZeros() - 1);
    for(size_t i = 0; i < 1024; ++i) {
        int    k = random_idx(rng);
        delta += fabs(v[k] - A.valuePtr()[k]);
    }

    std::cout
        << "delta = " << delta << std::endl
        << prof << std::endl;
}
