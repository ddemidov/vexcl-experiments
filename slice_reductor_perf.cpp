#include <iostream>
#include <vexcl/vexcl.hpp>

int main() {
    using vex::_;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    const size_t N = 64;
    const size_t M = 1024;

    vex::vector<double> x(ctx, N * N * N);

    vex::vector<double> y1(ctx, N);
    vex::vector<double> y2(ctx, N * N);

    std::vector<double> h1(N);
    std::vector<double> h2(N * N);

    x = vex::Random<double, vex::random::threefry>()(vex::element_index(), std::rand());

    vex::Reductor<double, vex::SUM> sum(ctx);
    vex::slicer<3> slice(vex::extents[N][N][N]);

    vex::profiler<> prof(ctx);

    std::array<size_t,2> reduce_dims = {{1, 2}};
    prof.tic_cl("slice reductor (1)");
    for(size_t i = 0; i < M; ++i)
        y1 = vex::reduce<vex::SUM>(slice[_](x), reduce_dims);
    prof.toc("slice reductor (1)");

    std::cout << y1[0] << " == ";

    prof.tic_cl("loop reductor (1)");
    for(size_t i = 0; i < M; ++i) {
        for(size_t j = 0; j < N; ++j)
            h1[j] = sum(slice[j](x));
        vex::copy(h1, y1);
    }
    prof.toc("loop reductor (1)");

    std::cout << y1[0] << std::endl;




    prof.tic_cl("slice reductor (2)");
    for(size_t i = 0; i < M; ++i)
        y2 = vex::reduce<vex::SUM>(slice[_](x), 2);
    prof.toc("slice reductor (2)");

    std::cout << y2[0] << " == ";

    prof.tic_cl("loop reductor (2)");
    for(size_t i = 0; i < M; ++i) {
        for(size_t j = 0, idx = 0; j < N; ++j)
            for(size_t k = 0; k < N; ++k, ++idx)
                h2[idx] = sum(slice[j][k](x));
        vex::copy(h2, y2);
    }
    prof.toc("loop reductor (2)");

    std::cout << y2[0] << std::endl;

    std::cout << prof << std::endl;
}
