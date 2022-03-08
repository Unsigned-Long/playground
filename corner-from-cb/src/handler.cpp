#include "handler.h"
#include <list>
#include <math.h>

namespace ns_test
{

#pragma region class Linefitting
    bool ns_test::CornerSelector::Linefitting::operator()(const double *const params, double *out) const
    {
        out[0] = params[0] * this->_p.x + params[1] * this->_p.y + 1.0;
        return true;
    }

#pragma endregion
#pragma region static values
    cv::Point2f CornerSelector::p1;
    cv::Point2f CornerSelector::p2;

    std::list<std::pair<cv::Point2f, bool>> CornerSelector::less;

    pcl::KdTreeFLANN<pcl::PointXY> CornerSelector::kdtree;

    cv::Mat CornerSelector::dst;

    std::vector<std::pair<cv::Point2f, bool>> CornerSelector::corners;

    std::vector<std::pair<cv::Point2f, cv::Point2f>> CornerSelector::mapping;

    int CornerSelector::rows;

    int CornerSelector::cols;

    float CornerSelector::blockSize;

    int CornerSelector::selected = 0;

    bool CornerSelector::isSorting = false;

#pragma endregion

#pragma region static functions

    cv::Point2f CornerSelector::fittingLine(const std::vector<cv::Point2f> &points)
    {
        double param[2] = {1.0, 1.0};
        ceres::Problem prob;
        for (const auto &p : points)
        {
            ceres::CostFunction *fun =
                new ceres::NumericDiffCostFunction<CornerSelector::Linefitting, ceres::CENTRAL, 1, 2>(new CornerSelector::Linefitting(p));
            prob.AddResidualBlock(fun, nullptr, param);
        }

        ceres::Solver::Options op;
        op.gradient_tolerance = 1E-20;
        op.function_tolerance = 1E-20;
        op.minimizer_progress_to_stdout = false;
        op.linear_solver_type = ceres::DENSE_QR;

        ceres::Solver::Summary s;
        ceres::Solve(op, &prob, &s);

        std::cout << "A : " << param[0] << " B : " << param[1] << std::endl;
        return cv::Point2f(param[0], param[1]);
    }

    double CornerSelector::distance(const cv::Point2f &lineParams, const cv::Point2f &p)
    {
        // Ax + By + 1.0 = 0.0
        auto up = std::abs(lineParams.x * p.x + lineParams.y * p.y + 1.0);
        auto down = std::sqrt(lineParams.x * lineParams.x + lineParams.y * lineParams.y);
        return up / down;
    }

    void CornerSelector::onMouse_select_corners(int event, int x, int y, int flags, void *userdata)
    {
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            p1.x = x;
            p1.y = y;
        }
        else if (event == cv::EVENT_LBUTTONUP)
        {
            p2.x = x;
            p2.y = y;
            cv::Rect2f r(p1, p2);
            std::cout << r << std::endl;
            int count = 0;
            for (auto &elem : CornerSelector::less)
            {
                if (r.contains(elem.first) && !elem.second)
                {
                    elem.second = true;
                    cv::circle(dst, elem.first, 10, cv::Scalar(0, 0, 255), 3);
                    ++count;
                }
            }
            selected += count;
            std::cout << "selected " << count << " new points" << std::endl;
            std::cout << "Now the selected corner points are " << selected << std::endl;
            cv::imshow("win", dst);
        }
        else if (event == cv::EVENT_RBUTTONDOWN)
        {
            std::cout << '[' << x << ',' << y << ']' << std::endl;
            int count = 0;
            auto threshold = 1000;
            for (auto &point : less)
            {
                if (!(point.second))
                    continue;
                auto dis_squre = std::pow(x - point.first.x, 2) + std::pow(y - point.first.y, 2);
                if (dis_squre < threshold)
                {
                    cv::circle(dst, point.first, 10, cv::Scalar(0, 255, 0), 3);
                    point.second = false;
                    ++count;
                }
            }
            selected -= count;
            std::cout << "clear " << count << " points" << std::endl;
            std::cout << "Now the selected corner points are " << selected << std::endl;
            cv::imshow("win", dst);
        }
    }

    float CornerSelector::distance(const cv::Point2f &lp1, const cv::Point2f &lp2, const cv::Point2f &p)
    {
        Eigen::Vector2f line(lp2.x - lp1.x, lp2.y - lp1.y);
        Eigen::Vector2f dir(p.x - lp1.x, p.y - lp1.y);
        auto test = line.dot(dir) / (line.norm() * dir.norm());
        auto angle = std::acos(line.dot(dir) / (line.norm() * dir.norm()));
        return dir.norm() * std::sin(angle);
    }

    void CornerSelector::onMouse_sort_corners(int event, int x, int y, int flags, void *userdata)
    {
        if (isSorting)
            return;
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            p1.x = x;
            p1.y = y;
        }
        else if (event == cv::EVENT_LBUTTONUP)
        {
            p2.x = x;
            p2.y = y;
            isSorting = true;
            // the points to fit line
            std::vector<cv::Point2f> vec{p1, p2};
            int count = 0;
            // fiting line for [rows] times
            while (count < rows)
            {
                auto line = ns_test::CornerSelector::fittingLine(vec);
                std::cout << "Fitted Line : " << line << std::endl;
                // sort the points by the distance
                std::sort(corners.begin(), corners.end(), [&line](const std::pair<cv::Point2f, bool> &com_p1, const std::pair<cv::Point2f, bool> &com_p2)
                          { return CornerSelector::distance(line, com_p1.first) <
                                   CornerSelector::distance(line, com_p2.first); });
                vec.clear();
                auto iter = corners.begin();
                // add the k-nearest points
                while (vec.size() != cols)
                {
                    if (iter->second == false)
                    {
                        vec.push_back(iter->first);
                        iter->second = true;
                    }
                    ++iter;
                }
                // sort the x
                std::sort(vec.begin(), vec.end(), [](const cv::Point2f &com_p1, const cv::Point2f &com_p2)
                          { return com_p1.x < com_p2.x; });
                // output
                for (auto i = vec.begin(); i != vec.end(); ++i)
                {
                    mapping.push_back(std::make_pair(*i, cv::Point2f((i - vec.begin()) * blockSize, count * blockSize)));
                    cv::drawMarker(dst, *i, cv::Scalar(0, 255, 255), 0, 50, 3);
                }
                ++count;
                // display
                cv::imshow("win", dst);
                cv::waitKey(1000);
            }
            cv::destroyAllWindows();
        }
    }

    void CornerSelector::doSearch(const std::string &imagePath, int rows, int cols, float realBlockSize, float threshold, float combineSize, bool isColor)
    {
        // assign
        CornerSelector::rows = rows;
        CornerSelector::cols = cols;
        CornerSelector::blockSize = realBlockSize;

        // read image and detect the init_corners
        std::cout << "load image..." << std::endl;
        std::vector<cv::Point> init_corners;

        // judge the output image type when operations below
        if (isColor)
        {
            dst = cv::imread(imagePath, cv::IMREAD_COLOR);
            cv::Mat temp, out;
            cv::cvtColor(dst, temp, cv::COLOR_RGB2GRAY);
            std::cout << "detect init_corners..." << std::endl;
            init_corners = ns_harris::Harris::cornerDetector(temp, out, 3, 0.05, threshold, ns_harris::Harris::Output::RVALUE);
        }
        else
        {
            auto img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
            init_corners = ns_harris::Harris::cornerDetector(img, dst, 3, 0.05, threshold, ns_harris::Harris::Output::RVALUE);
        }
        // decrease the points' number using kd-tree
        std::cout << "construct kdtree and decrease points' number..." << std::endl;
        pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
        for (const auto &elem : init_corners)
            cloud->push_back(pcl::PointXY(elem.x, elem.y));
        kdtree.setInputCloud(cloud);
        for (auto iter = cloud->begin(); iter != cloud->end(); ++iter)
        {
            if (iter->x < 0.0)
                continue;
            std::vector<int> index;
            std::vector<float> dis;
            // 20 is the range radius
            auto search = kdtree.radiusSearch(*iter, combineSize, index, dis);
            if (search > 0)
            {
                pcl::PointXY mean(0, 0);
                for (const auto &elem : index)
                {
                    mean.x += cloud->at(elem).x;
                    mean.y += cloud->at(elem).y;
                    cloud->at(elem).x = -1.0;
                }
                mean.x /= index.size();
                mean.y /= index.size();
                less.push_back(std::make_pair(cv::Point2f(mean.x, mean.y), false));
            }
        }
        std::cout << "points decrease from " << cloud->size() << " to " << less.size() << std::endl;

        // mark the init corner points on the image
        for (const auto &elem : less)
            cv::circle(dst, elem.first, 10, cv::Scalar(0, 255, 0), 3);

        // operations to select the true corner of the chessboard, using 'q' to quit the process
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "use mouse to select the true corners..." << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        cv::namedWindow("win", cv::WINDOW_FREERATIO);
        cv::setMouseCallback("win", CornerSelector::onMouse_select_corners, 0);
        cv::imshow("win", dst);
        cv::waitKey(0);

        // clear the points that don't selected
        for (const auto &elem : CornerSelector::less)
        {
            if (!elem.second)
                continue;
            CornerSelector::corners.push_back(std::make_pair(elem.first, false));
        }

        // operations to get the true corners
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "use mouse to draw lines at the beginning..." << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        cv::namedWindow("win", cv::WINDOW_FREERATIO);
        cv::setMouseCallback("win", CornerSelector::onMouse_sort_corners, 0);
        cv::imshow("win", dst);
        cv::waitKey(0);
    }

    std::vector<std::pair<cv::Point2f, cv::Point2f>> &CornerSelector::getMapping()
    {
        return mapping;
    }
#pragma endregion
} // namespace ns_test
