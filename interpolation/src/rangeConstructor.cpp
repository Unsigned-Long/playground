#include "rangeConstructor.h"

namespace ns_inter
{
    RangeConstructor::RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source, int size_x)
        : _src(source), _dst(new pcl::PointCloud<pcl::PointXYZI>())
    {
        auto min_max_x = std::minmax_element(source->cbegin(), source->cend(), [](const item_type &p1, const item_type &p2)
                                             { return p1.x < p2.x; });

        this->init(size_x, 1, 1,
                   min_max_x.first->x, 0.0, 0.0,
                   min_max_x.second->x, 0.0, 0.0);
    }
    RangeConstructor::RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source,
                                       int size_x, int size_y)
        : _src(source), _dst(new pcl::PointCloud<pcl::PointXYZI>())
    {
        auto min_max_x = std::minmax_element(source->cbegin(), source->cend(), [](const item_type &p1, const item_type &p2)
                                             { return p1.x < p2.x; });
        auto min_max_y = std::minmax_element(source->cbegin(), source->cend(), [](const item_type &p1, const item_type &p2)
                                             { return p1.y < p2.y; });
        this->init(size_x, size_y, 1,
                   min_max_x.first->x, min_max_y.first->y, 0.0,
                   min_max_x.second->x, min_max_y.second->y, 0.0);
    }

    RangeConstructor::RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source,
                                       int size_x, int size_y, int size_z)
        : _src(source), _dst(new pcl::PointCloud<pcl::PointXYZI>())

    {
        auto min_max_x = std::minmax_element(source->cbegin(), source->cend(), [](const item_type &p1, const item_type &p2)
                                             { return p1.x < p2.x; });
        auto min_max_y = std::minmax_element(source->cbegin(), source->cend(), [](const item_type &p1, const item_type &p2)
                                             { return p1.y < p2.y; });
        auto min_max_z = std::minmax_element(source->cbegin(), source->cend(), [](const item_type &p1, const item_type &p2)
                                             { return p1.z < p2.z; });
        this->init(size_x, size_y, size_z,
                   min_max_x.first->x, min_max_y.first->y, min_max_z.first->z,
                   min_max_x.second->x, min_max_y.second->y, min_max_z.second->z);
    }

    void RangeConstructor::init(int size_x, int size_y, int size_z,
                                float min_x, float min_y, float min_z,
                                float max_x, float max_y, float max_z)
    {
        if (min_x > max_x || min_y > max_y || min_z > max_z || size_x <= 0 || size_y <= 0 || size_z <= 0)
            throw std::invalid_argument("The argvs are invalid in the RangeConstructor[2d]!");
        auto step_x = size_x == 1 ? 0.0 : (max_x - min_x) / (size_x - 1);
        auto step_y = size_y == 1 ? 0.0 : (max_y - min_y) / (size_y - 1);
        auto step_z = size_z == 1 ? 0.0 : (max_z - min_z) / (size_z - 1);
        float cur_x = min_x;
        float cur_y = min_y;
        float cur_z = min_z;
        for (int i = 0; i != size_x; ++i)
        {
            for (int j = 0; j != size_y; ++j)
            {
                for (int k = 0; k != size_z; ++k)
                {
                    this->_dst->push_back(item_type(cur_x, cur_y, cur_z, 0.0));
                    cur_z += step_z;
                }
                cur_y += step_y;
                cur_z = min_z;
            }
            cur_x += step_x;
            cur_y = min_y;
        }
        return;
    }

} // namespace ns_inter
