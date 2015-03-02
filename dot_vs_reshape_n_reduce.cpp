#include <iostream>
#include <vexcl/vexcl.hpp>

int main() {
    const size_t N = 1024;
    const size_t M = 1024;
    const size_t L = 1024;

    // Test performance of dense matrix product for matrices
    // A[N][M] and B[M][L].

    vex::Context ctx(vex::Filter::Env && vex::Filter::DoublePrecision && vex::Filter::Count(1));
    std::cout << ctx << std::endl;

    vex::vector<double> A(ctx, N * M);
    vex::vector<double> B(ctx, M * L);
    vex::vector<double> C(ctx, N * L);

    A = vex::Random<double>()(vex::element_index(), 0);
    B = vex::Random<double>()(vex::element_index(), 1);

    vex::slicer<2> Adim(vex::extents[N][M]);
    vex::slicer<2> Bdim(vex::extents[M][L]);

    using vex::_;

#define PROD1                                                             \
    C = vex::reduce<vex::SUM>(                                            \
            vex::extents[N][M][L],                                        \
            vex::reshape(A, vex::extents[N][M][L], vex::extents[0][1]) *  \
            vex::reshape(B, vex::extents[N][M][L], vex::extents[1][2]),   \
            vex::extents[1]                                               \
            )

#define PROD2                                                             \
    C = vex::tensordot(Adim[_](A), Bdim[_](B), vex::axes_pairs(1, 0))

    PROD1;
    PROD2;

    vex::profiler<> prof(ctx);

    prof.tic_cl("reshape/reduce");
    for(int i = 0; i < 10; ++i) {
        PROD1;
    }
    prof.toc("reshape/reduce");

    prof.tic_cl("tensordot");
    for(int i = 0; i < 10; ++i) {
        PROD2;
    }
    prof.toc("tensordot");

    std::cout << prof << std::endl;
}
