#include "interIDW.h"

namespace ns_inter
{

    void InterIDW::interpolation(const RangeConstructor &rc, int nearestK, float power)
    {
        if (nearestK <= 0)
            throw std::invalid_argument("the nearestK must be greater than 0!");
        if (nearestK > rc._src->points.size())
            throw std::invalid_argument("the nearestK must be less equal than RangeConstructor.source's size!");
        power /= 2.0;
        pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
        kdtree.setInputCloud(rc._src);
        for (auto &target : rc._dst->points)
        {
            std::vector<int> index;
            std::vector<float> disSqure;
            auto count = kdtree.nearestKSearch(target, nearestK, index, disSqure);
            float v1 = 0.0;
            float v2 = 0.0;
            for (int i = 0; i != count; ++i)
            {
                auto val = rc._src->points.at(index.at(i)).intensity;
                auto weight = 1.0 / std::pow(disSqure.at(i), power);
                if (std::isinf(weight))
                    weight = MAXFLOAT / 2.0;
                v1 += val * weight;
                v2 += weight;
            }
            target.intensity = v1 / v2;
        }
    }

} // namespace ns_inter
