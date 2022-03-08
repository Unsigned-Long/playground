# A Cpp Class for Quaternion Calculation
>___Author : csl___  
>___E-mail : 3079625093@qq.com___

## 1.Overview

### Quaternion is a useful tool to express the rotation, and it's better than the rotation matrix.This cpp library supports the normal caculations of the quaternion, and especially it supports the rotation through a 3d vector as the main axis and a point. It's easy to use.
---
## Reference
>website : https://www.qiujiawei.com/understanding-quaternions/

## 3.Details

### Here is the head file and the methods.  
```cpp
namespace trans
{
#pragma region class Quaternion

    /**
     * \brief Implementation of lightweight C + + classes related to quaternion calculation
     */
    class Quaternion
    {
    public:
        // type define
        using value_type = double;
        using real_type = double;
        using imag_type = std::array<value_type, 3>;
        using vec_type = imag_type;

    private:
        // friend functions
        friend std::ostream &operator<<(std::ostream &os, const Quaternion &q);
        friend Quaternion::imag_type operator+(const imag_type &v1, const imag_type &v2);
        friend Quaternion::imag_type operator-(const imag_type &v1, const imag_type &v2);

    private:
        /**
         * \members
         * _real : the real part of the quaternion
         * _imag : the image part of the quaternion
         */
        real_type _real;
        imag_type _imag;

    public:
        // constructors
        Quaternion() : _real(0.0), _imag{0.0, 0.0, 0.0} {}
        Quaternion(real_type r, value_type i, value_type j, value_type k) : _real(r), _imag{i, j, k} {}
        Quaternion(real_type r, const imag_type &i) : _real(r), _imag(i) {}
        // deconstructor
        ~Quaternion() {}

        // get the real part of the quaternion
        const real_type &real() const;
        real_type &real();

        // get the image part of the quaternion
        const imag_type &imag() const;
        imag_type &imag();

        // Judge whether it is a real quaternion [!dangerous method]
        bool isRealQuaternion() const;

        // Judge whether it is a pure quaternion [!dangerous method]
        bool isPureQuaternion() const;

        /** splite the quaternion to a real and pure quaternion
         * \return [RealQuaternion, PureQuaternion]
         */
        std::pair<Quaternion, Quaternion> splite2RealPure() const;

        // Judge whether it is a unit quaternion [!dangerous method]
        bool isUnitQuaternion() const;

        // get the conjugate Quaternion and return it
        Quaternion getConjugateQuaternion() const;

        // get the Quaternion Norm
        value_type getNorm() const;

        // get the Normalized Quaternion and return it
        Quaternion getNormalizedQuaternion() const;

        // trans the quaternion to Normalized Quaternion return it's reference
        Quaternion &toNormalizedQuaternion();

        // get the Inverse Quaternion and return it
        Quaternion getInverseQuaternion() const;

    public:
        // add two quaternions
        static Quaternion add(const Quaternion &q1, const Quaternion &q2);

        // subtract two quaternions
        static Quaternion sub(const Quaternion &q1, const Quaternion &q2);

        // multiply two quaternions
        static Quaternion mul(const Quaternion &q1, const Quaternion &q2);

        // Multiplying a Quaternion by a Scalar
        static Quaternion mul(value_type v, const Quaternion &q);

        // Judge whether two quaternions are conjugates [!dangerous method]
        static bool isConjugateQuaternions(const Quaternion &q1, const Quaternion &q2);

        // caculate the dot of two quaternions
        static value_type dot(const Quaternion &q1, const Quaternion &q2);

        // get the Rotation Quaternion
        static Quaternion getRotationQuaternion(const vec_type &dir_vec, value_type angle_radian);

        static vec_type rotate(const Quaternion &rotationQuaternion, const vec_type &vec);

    private:
        /**
         * \brief some operations for vector
         */
        static vec_type
        vec_add(const vec_type &v1, const vec_type &v2);

        static vec_type vec_sub(const vec_type &v1, const vec_type &v2);

        static real_type vec_dot(const vec_type &v1, const vec_type &v2);

        static vec_type vec_mul(value_type v, const vec_type &i);

        static vec_type vec_cross(const vec_type &v1, const vec_type &v2);

        static value_type vec_module(const vec_type &v);

        static vec_type vec_Normalized(const vec_type &v);
    };
#pragma endregion

#pragma region outer helping functions
    // output the quaternion to ostream
    std::ostream &operator<<(std::ostream &os, const Quaternion &q);

    Quaternion operator+(const Quaternion &q1, const Quaternion &q2);

    Quaternion operator-(const Quaternion &q1, const Quaternion &q2);

    /**
     * \brief overload the opertaors for array [std::array<value_type, 3>]
     */
    std::ostream &operator<<(std::ostream &os, const Quaternion::vec_type &v);

    Quaternion::imag_type operator+(const Quaternion::imag_type &v1, const Quaternion::imag_type &v2);
    
    Quaternion::imag_type operator-(const Quaternion::imag_type &v1, const Quaternion::imag_type &v2);

#pragma endregion

    void foo();
} // namespace ns_quaternion
```

