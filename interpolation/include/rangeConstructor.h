#pragma once

#include <iostream>
#include "pcl-1.12/pcl/kdtree/kdtree_flann.h"
#include <vector>
#include <algorithm>

namespace ns_inter
{
    class InterIDW;

    class RangeConstructor
    {
        // the friend class[IDW : a interpolation method]
        friend class InterIDW;

    public:
        /**
         * \brief the point type is pcl::PointXYZI
         *        and the [x,y,z] is the position,
         *        the [i] is the reference value.
         *        if you want to do the one-dime interpolation, set [y,z] = [0.0,0.0].
         *        if you want to do the two-dime interpolation, set [z] = [0.0].
         */
        using item_type = pcl::PointXYZI;

    private:
        /**
         * \brief the _src is the source data to interpolation
         *        the _dst is points to be interpolated
         */
        pcl::PointCloud<item_type>::Ptr _src;
        pcl::PointCloud<item_type>::Ptr _dst;

    public:
        RangeConstructor() = delete;
        /**
         * \brief the source and the size in the direction
         * \attention it's for 1d data
         */
        RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source, int size_x);

        /**
         * \brief point the min and max of the range by yourself
         * \attention it's for 1d data
         */
        RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source,
                         int size_x, float min_x, float max_x)
            : _src(source), _dst(new pcl::PointCloud<pcl::PointXYZI>())
        {
            this->init(size_x, 1, 1,
                       min_x, 0.0, 0.0,
                       max_x, 0.0, 0.0);
        }

        /**
         * \brief the source and the size in two directions
         * \attention it's for 2d data
         */
        RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source, int size_x, int size_y);

        /**
         * \brief point the min and max of the range by yourself
         * \attention it's for 2d data
         */
        RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source,
                         int size_x, int size_y,
                         float min_x, float min_y,
                         float max_x, float max_y)
            : _src(source), _dst(new pcl::PointCloud<pcl::PointXYZI>())
        {
            this->init(size_x, size_y, 1,
                       min_x, min_y, 0.0,
                       max_x, max_y, 0.0);
        }

        /**
         * \brief the source and the size in three directions
         * \attention it's for 3d data
         */
        RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source, int size_x, int size_y, int size_z);

        /**
         * \brief point the min and max of the range by yourself
         * \attention it's for 3d data
         */
        RangeConstructor(const pcl::PointCloud<item_type>::Ptr &source,
                         int size_x, int size_y, int size_z,
                         float min_x, float min_y, float min_z,
                         float max_x, float max_y, float max_z)

            : _src(source), _dst(new pcl::PointCloud<pcl::PointXYZI>())
        {
            this->init(size_x, size_y, size_z,
                       min_x, min_y, min_z,
                       max_x, max_y, max_z);
        }

        /**
         * \brief get the range points array
         */
        const pcl::PointCloud<item_type>::Ptr &pointArray() const { return this->_dst; }

    private:
        /**
         * \brief a function to construct the range points array
         */
        void init(int size_x, int size_y, int size_z,
                  float min_x, float min_y, float min_z,
                  float max_x, float max_y, float max_z);
    };
} // namespace ns_inter
