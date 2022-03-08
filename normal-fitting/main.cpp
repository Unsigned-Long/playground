#include "normal_fitting.h"
#include <random>
#include <fstream>

int main(int argc, char const *argv[])
{
    std::fstream raw("../pyDrawer/file1.txt", std::ios::out);
    std::fstream result("../pyDrawer/file2.txt", std::ios::out);
    std::default_random_engine e;
    std::uniform_real_distribution<> u(-0.05, 0.05);
    // params
    double mean_x = 5.0, mean_y = 4.0, sigma_x = 1.0, sigma_y = 3.0, rou = 0.1, A = 100;
    // generate data
    std::vector<ns_point::Point3d> vec;
    for (double x = 0.0; x < 10.0; x += 0.5)
    {
        for (double y = 0.0; y < 10.0; y += 0.5)
        {
            auto z = A * ns_gaussian::Gaussian<double>::gaussian(x, y, mean_x, mean_y, sigma_x, sigma_y, rou) + u(e);
            raw << x << ',' << y << ',' << z << std::endl;
            vec.push_back(ns_point::Point3d(x, y, z));
        }
    }
    // fitting it
    ns_norm::NormalFit::two_normal_fitting(vec, mean_x, mean_y, sigma_x, sigma_y, rou, A);
    int count = 0;
    for (double x = 0.0; x < 10.0; x += 0.5)
    {
        for (double y = 0.0; y < 10.0; y += 0.5, ++count)
        {
            auto z = A * ns_gaussian::Gaussian<double>::gaussian(x, y, mean_x, mean_y, sigma_x, sigma_y, rou);
            result << x << ',' << y << ',' << vec.at(count).z() - z << std::endl;
        }
    }
    // output result
    std::cout << "mean x : " << mean_x << std::endl;
    std::cout << "mean y : " << mean_y << std::endl;
    std::cout << "sigma x : " << sigma_x << std::endl;
    std::cout << "sigma y : " << sigma_y << std::endl;
    std::cout << "rou : " << rou << std::endl;
    std::cout << "A : " << A << std::endl;
    raw.close();
    result.close();
    return 0;
}
