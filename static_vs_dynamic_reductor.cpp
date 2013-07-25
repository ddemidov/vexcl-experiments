#include <iostream>
#include <vexcl/vexcl.hpp>

template <class T>
T static_sum(const vex::vector<T> &x) {
    const vex::Reductor<T, vex::SUM> &sum =
        vex::get_reductor<T, vex::SUM>(x.queue_list());
    return sum(x);
}

template <class T>
T dynamic_sum(const vex::vector<T> &x) {
    vex::Reductor<T, vex::SUM> sum(x.queue_list());
    return sum(x);
}

int main() {
    vex::Context ctx( vex::Filter::Env );
    std::cout << ctx << std::endl;

    vex::profiler<> prof;

    const size_t N = 1024;
    const size_t M = 1024 * 16;

    vex::vector<double> x(ctx, N);
    x = vex::Random<double, vex::random::threefry>()(
            vex::element_index(), std::rand());

    std::cout << "static..." << std::flush;

    prof.tic_cpu("static");
    double s1 = 0;
    for(size_t i = 0; i < M; ++i)
        s1 += static_sum(x);
    prof.toc("static");

    std::cout << " done" << std::endl << "dynamic..." << std::flush;

    prof.tic_cpu("dynamic");
    double s2 = 0;
    for(size_t i = 0; i < M; ++i)
        s2 += dynamic_sum(x);
    prof.toc("dynamic");

    std::cout << " done" << std::endl
              << s1 << " == " << s2 << std::endl;

    std::cout << prof << std::endl;
}
