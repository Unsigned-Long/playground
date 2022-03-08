#pragma once

#include "rangeConstructor.h"
#include <cmath>

namespace ns_inter
{
    class InterIDW
    {
    public:
    private:
        InterIDW() = delete;

    public:
        /**
     * \brief a function to do the idw interpolation using nearestK search method to get points
     * \param rc the RangeConstructor contains the dst points array
     * \param nearestK the nearest points' number
     * \param power the param in the idw interpolation
     */ 
        static void interpolation(const RangeConstructor &rc, int nearestK, float power = 2.0);

    };
} // namespace ns_inter
