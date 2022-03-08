#include "interIDW.h"
#include <random>
#include <fstream>

std::vector<std::string> split(const std::string &str, char splitor)
{
    std::vector<std::string> vec;
    auto iter = str.cbegin();
    while (true)
    {
        auto pos = std::find(iter, str.cend(), splitor);
        vec.push_back(std::string(iter, pos));
        if (pos == str.cend())
            break;
        iter = ++pos;
    }
    return vec;
}

int main(int argc, char const *argv[])
{
    if (argc != 7)
    {
        std::cout << "Wrong Argvs" << std::endl;
        exit(-1);
    }
    /**
     * \brief 
     * argv : source data file name [1]
     *        output file name      [2]
     *        x range count         [3]
     *        y range count         [4]
     *        nearest K             [5]
     *        power                 [6]
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr source(new pcl::PointCloud<pcl::PointXYZI>());
    std::fstream dataFile(argv[1], std::ios::in);
    std::fstream outputFile(argv[2], std::ios::out);
    std::string strLine;
    while (std::getline(dataFile, strLine))
    {
        auto vec = split(strLine, ',');
        pcl::PointXYZI tempPos;
        tempPos.x = std::stod(vec[1]);
        tempPos.y = std::stod(vec[0]);
        tempPos.z = 0.0;
        tempPos.intensity = std::stod(vec[2]);
        source->points.push_back(tempPos);
    }
    dataFile.close();
    ns_inter::RangeConstructor rc(source, std::stoi(argv[3]), std::stoi(argv[4]));
    ns_inter::InterIDW::interpolation(rc, std::stoi(argv[5]), std::stod(argv[6]));

    for (const auto &elem : rc.pointArray()->points)
        outputFile << elem.y << ',' << elem.x << ',' << elem.intensity << std::endl;
    outputFile.close();
    return 0;
}
