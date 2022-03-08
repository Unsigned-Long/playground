# Camera-Calibration
>___Author : csl___  
>___E-nail : 3079625093@qq.com___

## OverView

### When we get a picture and identify it, how many pixels are the distance between the two parts, but how many pixels correspond to the meters in the real world? Therefore, it is necessary to use the results of camera calibration to convert the pixel coordinates to physical coordinates to calculate the distance (of course, it is worth explaining here that only using the results of monocular camera calibration can not directly convert the pixel coordinates to physical coordinates, because the perspective projection loses the coordinates of one dimension, so the ranging actually needs a binocular camera).This program is based on Zhang Zhengyou calibration method to realize measurement of the camera internal parameter distortion coefficient.

## Details 
>___imgs___  

<img src="./data/images/img_Result1.png" width="25%"><img src="./data/images/img_Result2.png" width="25%"><img src="./data/images/img_Result3.png" width="25%"><img src="./data/images/img_Result4.png" width="25%"><img src="./data/images/img_Result5.png" width="25%"><img src="./data/images/img_Result6.png" width="25%"><img src="./data/images/img_Result7.png" width="25%"><img src="./data/images/img_Result8.png" width="25%">  

## Result

```cpp
Load Data
---------
Calculate Params Matrix
-----------------------
Calculate Inner Params Matrix
-----------------------------
Calculate Outer Params Matrix
-----------------------------
Do Levenberg-Marquardt
----------------------
Calculate Distortion Params
---------------------------
Finished
--------
Inner Params Matrix
-------------------
 3.10354e+03 -4.84832e+00  2.00307e+03
 0.00000e+00  3.10291e+03  1.48657e+03
 0.00000e+00  0.00000e+00  1.00000e+00
Details
-------
   cx :  2.00307e+03
   cy :  1.48657e+03
   fx :  3.10354e+03
   fy :  3.10290e+03
Theta :  1.56923e+00
   K1 :  5.99269e-02
   K2 : -3.07124e-01
   K3 :  2.66638e-01
```

## Usage
```cpp
int main(int argc, char const *argv[])
{
    // load data
    ns_test::Camera camera("../data/mapping");
    // do the process
    camera.process();
    // get the params
    auto inner = camera.innerParamsMatrix();
    auto k_tuple = camera.distortionParams();
    
    return 0;
}
```

## Details

```cpp
    class Camera
    {
    private:
#pragma region inner structures
        /**
         * \brief a structure to organize data
         */
        struct Mapping
        {
            Point2d _pixel;
            Point2d _real;
            Mapping() = default;
            Mapping(const Point2d &pixel, const Point2d &real) : _pixel(pixel), _real(real) {}
        };
        /**
         * \brief a structure to organize data
         */
        struct ImgData
        {
            std::vector<Mapping> _mappingVec;

            Eigen::Matrix3d _H;

            Eigen::Matrix3d _Rt;

            ImgData() = default;
        };

        /**
         * \brief a ceres structure to calculate the params matrix
         */
        struct Ceres_Params
        {
            const Mapping *_map;

            Ceres_Params(const Mapping *map) : _map(map) {}
            /**
             * \brief params[8] out[2]
             */
            bool operator()(const double *const params, double *out) const;
        };

        /**
         * \brief a ceres structure to calculate the inner params matrix
         */
        struct Ceres_InnerParams
        {
            const Eigen::Matrix3d *_H;

            Ceres_InnerParams(const Eigen::Matrix3d *H) : _H(H) {}

            /**
             * \brief params[6] out[2]
             */
            bool operator()(const double *const params, double *out) const;

            Eigen::MatrixXd helper(int i, int j) const;
        };

        /**
         * \brief a ceres structure to do Levenberg-Marquardt
         */
        struct Ceres_ML
        {
            const Mapping *_map;

            Ceres_ML(const Mapping *map) : _map(map) {}
            /**
             * \brief params[5 + 6] out[2]
             */
            bool operator()(const double *const innerParams, const double *const outerParams, double *out) const;
        };

        /**
         * \brief a ceres structure to calculate the distortion Params
         */
        struct Ceres_DistortionParams
        {
            Point2d _disPixel;
            Point2d _idealPixel;
            Point2d _centerPixel;
            double _r;

            Ceres_DistortionParams(const Point2d &disPixel, const Point2d &idealPixel, const Point2d &centerPixel, double r)
                : _disPixel(disPixel), _idealPixel(idealPixel), _centerPixel(centerPixel), _r(r) {}
            /**
             * \brief params[2] out[2]
             */
            bool operator()(const double *const params, double *out) const;
        };
        
#pragma endregion
    private:
        std::vector<ImgData>
            _data;

        Eigen::Matrix3d _A;

        double _k1;
        double _k2;

        double _theta;

        double _fx;

        double _fy;

    public:
        Camera() = delete;

        /**
         * \brief to load the data
         */
        Camera(std::string mapping_data_dir) { this->init(mapping_data_dir); }

        /**
         * \brief get the data
         */
        const std::vector<ImgData> &data() const { return this->_data; }

        /**
         * \brief do the process
         */
        void process();

        /**
         * \brief get the distortion Params[k1, k2]
         */
        std::pair<double, double> distortionParams() { return std::make_pair(this->_k1, this->_k2); }

        /**
         * \brief get the inner Params matrix
         */
        const Eigen::Matrix3d &innerParamsMatrix() const { return this->_A; }

    private:
#pragma region private static methods
        /**
         * \brief split a string according the splitor
         */
        static std::vector<std::string> split(const std::string &str, char splitor);
        /**
         * \brief some static private methods
         */
        static void calParams(Camera::ImgData &imgData);

        static void calInitInnerParams(const std::vector<ImgData> &data, double params[6]);

        static Eigen::Matrix3d calOuterParams(const Eigen::Matrix3d &A, const Eigen::Matrix3d &H);

        static Eigen::Matrix3d organizeInnerMatrix(const double params[6]);

        static void doLM(const std::vector<ImgData> &data, double params[5]);

        static void output2Console(const std::string &str);

        static void calDistortionParams(const std::vector<ImgData> &data, const Eigen::Matrix3d &A, double params[2]);

        static void Optimization(const std::vector<ImgData> &data, double innerParams[5], double distortionParams[2]);
#pragma endregion

    private:
#pragma region private member methods
        /**
         * \brief init the class object
         */
        void init(std::string mapping_data_dir);

#pragma endregion
    };
```