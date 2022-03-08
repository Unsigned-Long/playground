#include "gaussian.hpp"
#include <vector>
#include <fstream>

int main(int argc, char const *argv[])
{
    std::fstream file("../pyDrawer/file4.txt", std::ios::out);
    // for (float x = 0.0; x < 10.0; x += 0.5)
    //     file << x << ',' << ns_gaussian::Gaussian<float>::gaussianNormalized(x, 5.0, 5.0) << std::endl;

    std::vector<double> vec;
    for (float x = 0.0; x < 10.0; x += 0.1)
        for (float y = 0.0; y < 5.0; y += 0.1)
            file << x << ',' << y << ',' << ns_gaussian::Gaussian<float>::gaussianNormalized(x, y, 6.0, 2.5, 2.5, 3.0, 0.6) << std::endl;

    file.close();
    return 0;
}
