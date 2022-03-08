#pragma once

#include <iostream>
#include <cmath>
#include <list>
#include <fstream>
#include <string>
#include <exception>
#include <algorithm>
#include <array>

namespace ns_point
{
#pragma region global
    constexpr double PI = 3.1415926535;

    template <typename _Ty>
    void writeBinaryData(const std::list<_Ty> &ls, std::ofstream &file)
    {
        for (const auto &elem : ls)
            file.write((const char *)(&elem), sizeof(_Ty));
        return;
    }

    template <typename _Ty>
    void readBinaryData(std::list<_Ty> &ls, std::ifstream &file)
    {
        _Ty elem;
        file.seekg(0, std::ios::end);
        auto size = file.tellg() / sizeof(_Ty);
        file.seekg(0, std::ios::beg);
        int count = 0;
        while (!file.eof() && count < size)
        {
            file.read((char *)(&elem), sizeof(_Ty));
            ls.push_back(elem);
            ++count;
        }
    }
#pragma region global for Point2 < _Ty>
    template <typename _Ty>
    class Point2;

    using Point2f = Point2<float>;
    using Point2d = Point2<double>;
    using Point2i = Point2<int>;

    template <typename _Ty>
    float distance(const Point2<_Ty> &p1, const Point2<_Ty> &p2)
    {
        return static_cast<float>(std::sqrt(std::pow(p1.x() - p2.x(), 2) + std::pow(p1.y() - p2.y(), 2)));
    }

    template <typename _Ty>
    float azimuth(const Point2<_Ty> &from, const Point2<_Ty> &to)
    {
        auto deta_x = to.x() - from.x();
        auto deta_y = to.y() - from.y();
        auto azi = std::atan2(deta_y, deta_x);
        if (deta_y < 0.0f)
            azi += 2.0f * PI;
        return azi;
    }

    template <typename _Ty>
    void writePoints(const std::list<Point2<_Ty>> &points, const std::string &filePath, std::ios_base::openmode mode = std::ios::out | std::ios::binary)
    {
        std::ofstream file(filePath, mode);
        if (!file.is_open())
            throw std::ios_base::failure("File Open Failed");
        if (std::ios::binary == (mode & std::ios::binary))
        {
            ns_point::writeBinaryData(points, file);
        }
        else
        {
            for (const auto &point : points)
            {
                file << point.x() << ',' << point.y() << '\n';
            }
        }
    }

    template <typename _Ty>
    void readPoints(std::list<Point2<_Ty>> &points, const std::string &filePath, std::ios_base::openmode mode = std::ios::in | std::ios::binary)
    {
        std::ifstream file(filePath, mode);
        if (!file.is_open())
            throw std::ios_base::failure("File Open Failed");
        if (std::ios::binary == (mode & std::ios::binary))
        {
            ns_point::readBinaryData(points, file);
        }
        else
        {
            Point2<_Ty> point;
            std::string str;
            while (!file.eof())
            {
                std::getline(file, str);
                if (str.empty())
                    continue;
                auto iter = std::find(str.cbegin(), str.cend(), ',');
                point.x() = static_cast<_Ty>(std::stod(std::string(str.cbegin(), iter)));
                point.y() = static_cast<_Ty>(std::stod(std::string(++iter, str.cend())));
                points.push_back(point);
            }
        }
    }

#pragma endregion

#pragma region global for Point3 < _Ty>

    template <typename _Ty>
    class Point3;

    using Point3f = Point3<float>;
    using Point3d = Point3<double>;
    using Point3i = Point3<int>;

    template <typename _Ty>
    float distance(const Point3<_Ty> &p1, const Point3<_Ty> &p2)
    {
        return static_cast<float>(std::sqrt(std::pow(p1.x() - p2.x(), 2) + std::pow(p1.y() - p2.y(), 2) + std::pow(p1.z() - p2.z(), 2)));
    }

    template <typename _Ty>
    void writePoints(const std::list<Point3<_Ty>> &points, const std::string &filePath, std::ios_base::openmode mode = std::ios::out | std::ios::binary)
    {
        std::ofstream file(filePath, mode);
        if (!file.is_open())
            throw std::ios_base::failure("File Open Failed");
        if (std::ios::binary == (mode & std::ios::binary))
        {
            ns_point::writeBinaryData(points, file);
        }
        else
        {
            for (const auto &point : points)
            {
                file << point.x() << ',' << point.y() << ',' << point.z() << '\n';
            }
        }
    }

    template <typename _Ty>
    void readPoints(std::list<Point3<_Ty>> &points, const std::string &filePath, std::ios_base::openmode mode = std::ios::in | std::ios::binary)
    {
        std::ifstream file(filePath, mode);
        if (!file.is_open())
            throw std::ios_base::failure("File Open Failed");
        if (std::ios::binary == (mode & std::ios::binary))
        {
            ns_point::readBinaryData(points, file);
        }
        else
        {
            Point3<_Ty> point;
            std::string str;
            while (!file.eof())
            {
                std::getline(file, str);
                if (str.empty())
                    continue;
                auto iter = std::find(str.cbegin(), str.cend(), ',');
                point.x() = static_cast<_Ty>(std::stod(std::string(str.cbegin(), iter)));
                auto iter2 = std::find(++iter, str.cend(), ',');
                point.y() = static_cast<_Ty>(std::stod(std::string(iter, iter2)));
                point.z() = static_cast<_Ty>(std::stod(std::string(++iter2, str.cend())));
                points.push_back(point);
            }
        }
    }

#pragma endregion

#pragma endregion

#pragma region class Point2 < _Ty>
    template <typename _Ty = float>
    class Point2
    {
    public:
        using value_type = _Ty;
        using ary_type = std::array<_Ty, 2>;

    private:
        value_type _x;
        value_type _y;

    public:
        Point2() = default;
        Point2(value_type x, value_type y) : _x(x), _y(y) {}
        Point2(const ary_type &p) : _x(p[0]), _y(p[1]) {}
        operator ary_type() const;
        value_type &x();
        value_type &y();
        const value_type &x() const;
        const value_type &y() const;
        ~Point2() {}
    };

    template <typename _Ty>
    std::ostream &operator<<(std::ostream &os, const Point2<_Ty> &p)
    {
        os << '[' << p.x() << ',' << p.y() << ']';
        return os;
    }

    template <typename _Ty>
    Point2<_Ty>::operator ary_type() const
    {
        return ary_type{this->_x, this->_y};
    }

    template <typename _Ty>
    _Ty &Point2<_Ty>::x()
    {
        return this->_x;
    }

    template <typename _Ty>
    _Ty &Point2<_Ty>::y()
    {
        return this->_y;
    }

    template <typename _Ty>
    const _Ty &Point2<_Ty>::x() const
    {
        return this->_x;
    }

    template <typename _Ty>
    const _Ty &Point2<_Ty>::y() const
    {
        return this->_y;
    }
#pragma endregion

#pragma region class Point3 < _Ty>
    template <typename _Ty = float>
    class Point3
    {
    public:
        using value_type = _Ty;
        using ary_type = std::array<_Ty, 3>;

    private:
        value_type _x;
        value_type _y;
        value_type _z;

    public:
        Point3() = default;
        Point3(value_type x, value_type y, value_type z) : _x(x), _y(y), _z(z) {}
        Point3(const ary_type &p) : _x(p[0]), _y(p[1]), _z(p[2]) {}
        operator ary_type() const;
        value_type &x();
        value_type &y();
        value_type &z();
        const value_type &x() const;
        const value_type &y() const;
        const value_type &z() const;
        ~Point3() {}
    };

    template <typename _Ty>
    std::ostream &operator<<(std::ostream &os, const Point3<_Ty> &p)
    {
        os << '[' << p.x() << ',' << p.y() << ',' << p.z() << ']';
        return os;
    }

    template <typename _Ty>
    _Ty &Point3<_Ty>::x()
    {
        return this->_x;
    }

    template <typename _Ty>
    Point3<_Ty>::operator ary_type() const
    {
        return ary_type{this->_x, this->_y, this->_z};
    }

    template <typename _Ty>
    _Ty &Point3<_Ty>::y()
    {
        return this->_y;
    }

    template <typename _Ty>
    _Ty &Point3<_Ty>::z()
    {
        return this->_z;
    }

    template <typename _Ty>
    const _Ty &Point3<_Ty>::x() const
    {
        return this->_x;
    }

    template <typename _Ty>
    const _Ty &Point3<_Ty>::y() const
    {
        return this->_y;
    }

    template <typename _Ty>
    const _Ty &Point3<_Ty>::z() const
    {
        return this->_z;
    }
#pragma endregion
} // namespace ns_point
