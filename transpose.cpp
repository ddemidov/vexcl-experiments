#include <vector>
#include <vexcl/vexcl.hpp>

int main(int argc, char **argv) {
    const size_t n = argc > 1 ? std::stoi(argv[1]) : 1 << 20;
    const size_t m = argc > 2 ? std::stoi(argv[2]) : 4;

    std::vector<std::vector<double>> data;
    data.reserve(m);
    for(size_t i = 0; i < m; ++i) data.emplace_back(n, 1.0);

    vex::Context ctx(vex::Filter::Env);
    std::cout << ctx << std::endl;

    std::cout << "Size: " << n << "x" << m << std::endl;

    vex::profiler<> prof(ctx);

    vex::vector<double> dst(ctx, m * n);
    dst = 0;

    for(int k = 0; k < 5; ++k) {
        // 1. Do transpose on host, copy result to GPU.
        prof.tic_cl("Host-side transpose");
        {
            std::vector<double> buf(n * m);
            for(size_t i = 0; i < n; ++i)
                for(size_t j = 0; j < m; ++j)
                    buf[i * m + j] = data[j][i];

            vex::copy(buf, dst);
        }
        prof.toc("Host-side transpose");

        // 2. Do transpose on GPU.
        prof.tic_cl("GPU-side transpose");
        {
            vex::vector<double> buf(ctx, n * m);
            for(size_t j = 0; j < m; ++j)
                vex::copy(data[j].begin(), data[j].end(), dst.begin() + j * n);

            dst = vex::reshape(buf, vex::extents[n][m], vex::extents[1][0]);
        }
        prof.toc("GPU-side transpose");
    }

    std::cout << prof << std::endl;
}
