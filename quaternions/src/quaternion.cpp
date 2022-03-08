#include "quaternion.h"
#include "math.h"

namespace ns_trans
{

#pragma region class Quaternion

#pragma region member functions
    const Quaternion::real_type &Quaternion::real() const
    {
        return this->_real;
    }

    Quaternion::real_type &Quaternion::real()
    {
        return this->_real;
    }

    const Quaternion::imag_type &Quaternion::imag() const
    {
        return this->_imag;
    }

    Quaternion::imag_type &Quaternion::imag()
    {
        return this->_imag;
    }

    bool Quaternion::isRealQuaternion() const
    {
        std::equal_to<value_type> comp;
        return comp(this->_imag[0], 0.0) && comp(this->_imag[1], 0.0) && comp(this->_imag[2], 0.0);
    }

    bool Quaternion::isPureQuaternion() const
    {
        return std::equal_to<value_type>()(this->_real, 0.0);
    }

    std::pair<Quaternion, Quaternion> Quaternion::splite2RealPure() const
    {
        return std::make_pair(Quaternion(this->_real, imag_type{0.0, 0.0, 0.0}), Quaternion(0.0, this->_imag));
    }

    bool Quaternion::isUnitQuaternion() const
    {
        std::equal_to<value_type> comp;
        return comp(this->getNorm(), 1.0);
    }

    Quaternion Quaternion::getConjugateQuaternion() const
    {
        return Quaternion(this->_real, Quaternion::vec_mul(-1.0, this->_imag));
    }

    Quaternion::value_type Quaternion::getNorm() const
    {
        return std::sqrt(std::pow(this->_real, 2) +
                         std::pow(this->_imag[0], 2) +
                         std::pow(this->_imag[1], 2) +
                         std::pow(this->_imag[2], 2));
    }

    Quaternion Quaternion::getNormalizedQuaternion() const
    {
        return Quaternion::mul(1.0 / this->getNorm(), *this);
    }

    Quaternion &Quaternion::toNormalizedQuaternion()
    {
        *this = this->getNormalizedQuaternion();
        return *this;
    }

    Quaternion Quaternion::getInverseQuaternion() const
    {
        return Quaternion::mul(1.0 / std::pow(this->getNorm(), 2),
                               this->getConjugateQuaternion());
    }

#pragma endregion

#pragma region static functions
    Quaternion Quaternion::add(const Quaternion &q1, const Quaternion &q2)
    {
        Quaternion q = q1;
        q.real() += q2.real();
        for (int i = 0; i != 3; ++i)
            q.imag()[i] += q2.imag()[i];
        return q;
    }

    Quaternion Quaternion::sub(const Quaternion &q1, const Quaternion &q2)
    {
        Quaternion q = q1;
        q.real() -= q2.real();
        for (int i = 0; i != 3; ++i)
            q.imag()[i] -= q2.imag()[i];
        return q;
    }

    Quaternion Quaternion::mul(const Quaternion &q1, const Quaternion &q2)
    {
        auto &q1_real = q1.real();
        auto &q1_imag = q1.imag();
        auto &q2_real = q2.real();
        auto &q2_imag = q2.imag();
        return Quaternion(q1_real * q2_real - Quaternion::vec_dot(q1_imag, q2_imag),
                          Quaternion::vec_mul(q1_real, q2_imag) + Quaternion::vec_mul(q2_real, q1_imag) + Quaternion::vec_cross(q1_imag, q2_imag));
    }

    Quaternion Quaternion::mul(value_type v, const Quaternion &q)
    {
        return Quaternion(v * q._real, Quaternion::vec_mul(v, q._imag));
    }

    bool Quaternion::isConjugateQuaternions(const Quaternion &q1, const Quaternion &q2)
    {
        return std::equal_to<value_type>()(q1._real, q2._real) &&
               std::equal_to<imag_type>()(q1._imag, Quaternion::vec_mul(-1.0, q2._imag));
    }

    Quaternion::value_type Quaternion::dot(const Quaternion &q1, const Quaternion &q2)
    {
        return q1._real * q2._real + Quaternion::vec_dot(q1._imag, q2._imag);
    }

    Quaternion Quaternion::getRotationQuaternion(const vec_type &dir_vec, value_type angle_radian)
    {
        auto _cos = std::cos(angle_radian / 2.0);
        auto _sin = std::sin(angle_radian / 2.0);
        return Quaternion(_cos, Quaternion::vec_mul(_sin, Quaternion::vec_Normalized(dir_vec)));
    }

    Quaternion::vec_type Quaternion::rotate(const Quaternion &rotationQuaternion, const vec_type &vec)
    {
        return Quaternion::mul(Quaternion::mul(rotationQuaternion, Quaternion(0.0, vec)),
                               rotationQuaternion.getInverseQuaternion())
            ._imag;
    }
    ns_point::Point3d Quaternion::rotate(const Quaternion &rotationQuaternion, const ns_point::Point3d &p)
    {
        auto vec = Quaternion::mul(Quaternion::mul(rotationQuaternion, Quaternion(0.0, static_cast<ns_point::Point3d::ary_type>(p))),
                                   rotationQuaternion.getInverseQuaternion())
                       ._imag;
        return ns_point::Point3d(vec[0], vec[1], vec[2]);
    }
    Quaternion::vec_type Quaternion::vec_add(const vec_type &v1, const vec_type &v2)
    {
        return vec_type{v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]};
    }

    Quaternion::vec_type Quaternion::vec_sub(const vec_type &v1, const vec_type &v2)
    {
        return vec_type{v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]};
    }

    Quaternion::real_type Quaternion::vec_dot(const vec_type &v1, const vec_type &v2)
    {
        return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    }

    Quaternion::vec_type Quaternion::vec_mul(value_type v, const vec_type &i)
    {
        return vec_type{i[0] * v, i[1] * v, i[2] * v};
    }

    Quaternion::vec_type Quaternion::vec_cross(const vec_type &v1, const vec_type &v2)
    {
        return vec_type{v1[1] * v2[2] - v2[1] * v1[2], v1[2] * v2[0] - v2[2] * v1[0], v1[0] * v2[1] - v2[0] * v1[1]};
    }

    Quaternion::value_type Quaternion::vec_module(const vec_type &v)
    {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    Quaternion::vec_type Quaternion::vec_Normalized(const vec_type &v)
    {
        return Quaternion::vec_mul(1.0 / Quaternion::vec_module(v), v);
    }
#pragma endregion

#pragma endregion

#pragma region outer helping functions
    std::ostream &operator<<(std::ostream &os, const Quaternion &q)
    {
        os << '[' << q._real << ',' << q._imag[0] << ',' << q._imag[1] << ',' << q._imag[2] << ']';
        return os;
    }

    Quaternion operator+(const Quaternion &q1, const Quaternion &q2)
    {
        return Quaternion::add(q1, q2);
    }

    Quaternion operator-(const Quaternion &q1, const Quaternion &q2)
    {
        return Quaternion::sub(q1, q2);
    }

    std::ostream &operator<<(std::ostream &os, const Quaternion::vec_type &v)
    {
        os << '[' << v[0] << ',' << v[1] << ',' << v[2] << ']';
        return os;
    }

    Quaternion::vec_type operator+(const Quaternion::vec_type &v1, const Quaternion::vec_type &v2)
    {
        return Quaternion::vec_add(v1, v2);
    }

    Quaternion::vec_type operator-(const Quaternion::vec_type &v1, const Quaternion::vec_type &v2)
    {
        return Quaternion::vec_sub(v1, v2);
    }
#pragma endregion

} // namespace ns_quaternion