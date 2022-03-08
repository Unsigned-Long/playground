#include "handler.h"
#include <fstream>

int main(int argc, char const *argv[])
{
    ns_test::CornerSelector::doSearch("../images/img4.jpg", 6, 11, 0.022, 1E09, 10, false);
    auto &mapping = ns_test::CornerSelector::getMapping();
    /**
     * \brief for output, please Release annotation
     */
    std::ofstream file("../result/img4_mapping.txt", std::ios::out);
    for (const auto &elem : mapping)
        file << elem.first.x << ',' << elem.first.y << ',' << elem.second.x << ',' << elem.second.y << std::endl;
    file.close();
    return 0;
}
