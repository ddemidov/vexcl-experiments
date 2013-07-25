#include <iostream>
#include <vexcl/vexcl.hpp>

int main() {
    using vex::_;

    vex::Context ctx( vex::Filter::Env && vex::Filter::Count(1) );
    std::cout << ctx << std::endl;

    const size_t N = 64;
    const size_t M = 1024;

    vex::vector<double> x(ctx, N * N * N);
    vex::vector<double> y(ctx, N);
    std::vector<double> h(N);

    x = vex::Random<double, vex::random::threefry>()(vex::element_index(), std::rand());

    vex::profiler<> prof(ctx);

    vex::slicer<3> slice(vex::extents[N][N][N]);

    std::array<size_t,2> reduce_dims = {{1, 2}};
    prof.tic_cl("slice reductor");
    for(size_t i = 0; i < M; ++i)
        y = vex::reduce<vex::SUM>(slice[_](x), reduce_dims);
    prof.toc("slice reductor");

    std::cout << y[0] << " == ";

    vex::Reductor<double, vex::SUM> sum(ctx);

    prof.tic_cl("loop reductor");
    for(size_t i = 0; i < M; ++i) {
        for(size_t j = 0; j < N; ++j)
            h[j] = sum(slice[j](x));
        vex::copy(h, y);
    }
    prof.toc("loop reductor");

    std::cout << y[0] << std::endl;

    std::cout << prof << std::endl;
}
