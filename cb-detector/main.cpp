#include "handler.h"
#include <fstream>

int main(int argc, char const *argv[])
{
    ns_test::CBDector detector("../test/test0/8.jpg");
    detector.process(0.0118, "../test/test0/process");
    auto mapping = detector.getMapping();
    std::fstream file("../test/test0/output8.txt", std::ios::out);
    for (const auto &elem : mapping)
        file << elem._pixel.x << ',' << elem._pixel.y << ',' << elem._real.x << ',' << elem._real.y << std::endl;
    file.close();
    return 0;
}
