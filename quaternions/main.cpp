#include "quaternion.h"

using namespace ns_trans;
using namespace ns_point;

void foo()
{
    auto r = Quaternion::getRotationQuaternion({0, 0, 1}, 45.0 * M_PI / 180.0);
    Quaternion::vec_type point{2, 0, 0};
    auto result = Quaternion::rotate(r, point);
    std::cout << result << std::endl;
    Quaternion q(1.0, static_cast<Point3d::ary_type>(Point3d(1, 2, 3)));
    std::cout << q << std::endl;
    std::cout << "-----------" << std::endl;
    ns_point::Point3d p(2, 0, 0);
    auto result_p = Quaternion::rotate(Quaternion::getRotationQuaternion(Quaternion::vec_type{0, 0, 100}, 45.0 * M_PI / 180.0), p);
    std::cout << result_p << std::endl;
    return;
}

int main(int argc, char const *argv[])
{
    ::foo();
    return 0;
}
