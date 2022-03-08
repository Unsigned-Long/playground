#include "handler.h"

namespace ns_test
{

#pragma region inner struct chessBoard
    const cv::Point2f &CBDector::ChessBoard::operator()(int r, int c) const
    {
        return this->_corners.at(r * _cols + c);
    }

    void CBDector::ChessBoard::addRowUp()
    {
        ++_rows;
        auto iter = this->_corners.begin();
        for (int i = 0; i != _cols; ++i)
            iter = this->_corners.insert(iter, cv::Point2f());
    }

    void CBDector::ChessBoard::addRowDown()
    {
        ++_rows;
        for (int i = 0; i != _cols; ++i)
            this->_corners.push_back(cv::Point2f());
    }

    void CBDector::ChessBoard::addColLeft()
    {
        ++_cols;
        auto iter = this->_corners.begin();
        for (int i = 0; i != _rows; ++i, iter += _cols)
        {
            iter = this->_corners.insert(iter, cv::Point2f());
        }
    }

    void CBDector::ChessBoard::addColRight()
    {
        auto iter = this->_corners.begin() + _cols;
        ++_cols;
        for (int i = 0; i != _rows; ++i, iter += _cols)
        {
            iter = this->_corners.insert(iter, cv::Point2f());
        }
    }

    cv::Point2f &CBDector::ChessBoard::operator()(int r, int c)
    {
        return this->_corners.at(r * _cols + c);
    }

#pragma endregion

#pragma region private static methods
    float CBDector::direction(const cv::Point2f &start, const cv::Point2f &end)
    {
        auto dir = std::atan2(end.y - start.y, end.x - start.x);
        if (end.y - start.y < 0.0)
            dir += 2 * M_PI;
        return dir;
    }

    bool CBDector::isInRange(float start_radian, float end_radian, float target)
    {
        if (start_radian < end_radian)
            return target > start_radian && target < end_radian;
        else
            return target > start_radian || target < end_radian;
    }

    float CBDector::distance(const cv::Point2f &p1, const cv::Point2f &p2)
    {
        return std::sqrt(std::pow(p1.x - p2.x, 2) + std::pow(p1.y - p2.y, 2));
    }

    cv::Mat CBDector::createCornerPrototype(float start_radian, float end_radian, int size)
    {
        cv::Mat img(size, size, CV_8UC1, cv::Scalar(0));
        auto halfSize = (size - 1) / 2;
        cv::Point2f start(halfSize, -halfSize);
        for (int i = 0; i != img.rows; ++i)
        {
            auto ptr = img.ptr<uchar>(i);
            for (int j = 0; j != img.cols; ++j)
            {
                cv::Point2f end(j, -i);
                auto dis = CBDector::distance(cv::Point2f(i, j), cv::Point2f(halfSize, halfSize));
                if (CBDector::isInRange(start_radian, end_radian, CBDector::direction(start, end)))
                    ptr[j] = 255 * Gaussian<float>::gaussianNormalized(dis, 0, size / 4.5);
            }
        }
        return img;
    }

    void CBDector::outputInConsole(const std::string &words)
    {
        std::string line(words.size(), '-');
        std::cout << std::string(words.size(), '-') << std::endl
                  << words << std::endl;
        return;
    }

    std::pair<int, int> CBDector::findModesMeanShift(const std::array<float, 32> &hist)
    {
        std::vector<int> info;
        auto gi = [](int i)
        {
            if (i < 0)
                return i + 32;
            if (i > 31)
                return i - 32;
            return i;
        };
        for (int i = 0; i < 32;)
        {

            if (hist.at(gi(i)) > hist.at(gi(i + 1)))
            {
                if (hist.at(gi(i)) >= hist.at(gi(i - 1)))
                    info.push_back(gi(i));
            }
            else
            {
                i++;
                while (i < 32 && hist.at(gi(i)) <= hist.at(gi(i + 1)))
                    ++i;
                if (i < 32)
                    info.push_back(gi(i));
            }
            i += 2;
        }
        std::sort(info.begin(), info.end(), [&hist](int i, int j)
                  { return hist.at(i) > hist.at(j); });
        return std::make_pair(info.at(0), info.at(1));
    }

    cv::Vec2f CBDector::gradVector(float grad)
    {
        auto x = std::cos(grad);
        auto y = std::sin(grad);
        cv::Vec2f vec(x, y);
        cv::normalize(vec);
        return vec;
    }
#pragma endregion

#pragma region private member methods
    void CBDector::init()
    {
        CBDector::outputInConsole("Init Gray Image and Create Prototypes");
        // get the gray image
        cv::cvtColor(this->_imgSrc, this->_imgGray, cv::COLOR_BGR2GRAY);
        // decide the size of the corner prototypes by the size of the image
        int size = ns_test::params::initProtoSize;
        int type1 = 0, type2 = 45, halfsize = (size - 1) / 2;
        // create the prototypes
        for (int i = 0; i != 4; ++i, type1 += 90, type2 += 90)
        {
            this->_type1.at(i) = this->createCornerPrototype(type1 * M_PI / 180.0, (type1 + 90) * M_PI / 180.0, size);
            this->_type2.at(i) = this->createCornerPrototype(type2 * M_PI / 180.0, (type2 + 90) % 360 * M_PI / 180.0, size);
        }
        this->_type2.back().ptr<uchar>(halfsize)[halfsize] = 0;
        CBDector::outputInConsole("Init Finished");
        return;
    }

    void CBDector::calCornerLikehood()
    {
        CBDector::outputInConsole("Calculate the Corner Likehood");
        cv::Mat temp(this->_imgGray.size(), CV_32FC1);
        auto ProtoSize = this->_type1.front().rows;
        auto halfProtoSize = (ProtoSize - 1) / 2;
        for (int i = halfProtoSize; i != this->_imgGray.rows - halfProtoSize; ++i)
        {
            auto ptrLikehood = temp.ptr<_Float32>(i);
            for (int j = halfProtoSize; j != this->_imgGray.cols - halfProtoSize; ++j)
            {
                // for the convolution
                int protoRowCount = 0, protoColCount = 0;
                int fType1A = 0, fType1B = 0, fType1C = 0, fType1D = 0;
                int fType2A = 0, fType2B = 0, fType2C = 0, fType2D = 0;
                for (int k = i - halfProtoSize; k != i + halfProtoSize + 1; ++k, ++protoRowCount)
                {
                    // for type 1
                    auto ptrType1A = this->_type1.at(0).ptr<uchar>(protoRowCount);
                    auto ptrType1C = this->_type1.at(1).ptr<uchar>(protoRowCount);
                    auto ptrType1B = this->_type1.at(2).ptr<uchar>(protoRowCount);
                    auto ptrType1D = this->_type1.at(3).ptr<uchar>(protoRowCount);
                    // for type 2
                    auto ptrType2A = this->_type2.at(0).ptr<uchar>(protoRowCount);
                    auto ptrType2D = this->_type2.at(1).ptr<uchar>(protoRowCount);
                    auto ptrType2B = this->_type2.at(2).ptr<uchar>(protoRowCount);
                    auto ptrType2C = this->_type2.at(3).ptr<uchar>(protoRowCount);
                    // for gray image
                    auto ptrGray = this->_imgGray.ptr<uchar>(k);
                    for (int l = j - halfProtoSize; l != j + halfProtoSize + 1; ++l, ++protoColCount)
                    {
                        // for type 1
                        fType1A += ptrGray[l] * ptrType1A[protoColCount];
                        fType1B += ptrGray[l] * ptrType1B[protoColCount];
                        fType1C += ptrGray[l] * ptrType1C[protoColCount];
                        fType1D += ptrGray[l] * ptrType1D[protoColCount];
                        // for type 2
                        fType2A += ptrGray[l] * ptrType2A[protoColCount];
                        fType2B += ptrGray[l] * ptrType2B[protoColCount];
                        fType2C += ptrGray[l] * ptrType2C[protoColCount];
                        fType2D += ptrGray[l] * ptrType2D[protoColCount];
                    }
                    // rollback
                    protoColCount = 0;
                }
                // for type 1
                auto mu1 = 0.25 * (fType1A + fType1B + fType1C + fType1D);
                auto s1Type1 = std::min(std::min(fType1A, fType1B) - mu1, mu1 - std::min(fType1C, fType1D));
                auto s2Type1 = std::min(mu1 - std::min(fType1A, fType1B), std::min(fType1C, fType1D) - mu1);
                // for type 2
                auto mu2 = 0.25 * (fType2A + fType2B + fType2C + fType2D);
                auto s1Type2 = std::min(std::min(fType2A, fType2B) - mu2, mu2 - std::min(fType2C, fType2D));
                auto s2Type2 = std::min(mu2 - std::min(fType2A, fType2B), std::min(fType2C, fType2D) - mu2);
                // calculate c
                auto c = std::max({s1Type1, s2Type1, s1Type2, s2Type2});
                // assign
                ptrLikehood[j] = c > 0.0 ? c : 0.0;
            }
        }
        this->_imgLikehood.create(this->_imgGray.size(), CV_8UC1);
        cv::normalize(temp, this->_imgLikehood, 255, 0, cv::NORM_MINMAX, 0);
        /**
         * \brief the value of 150 is the low boundary for the chessboard corner candidates,
         *        larger value means find less candidates.
         *        Change it if necessary.
         */
        cv::threshold(this->_imgLikehood, this->_imgLikehood, ns_test::params::likeHoodThreshold, 255, cv::ThresholdTypes::THRESH_TOZERO);
        CBDector::outputInConsole("Calculate Finished");
        return;
    }

    void CBDector::nonMaximumSuppression()
    {
        CBDector::outputInConsole("Non Maximum Suppression");
        constexpr int blockSize = 3;
        // [x, y]
        auto halfSize = (blockSize - 1) / 2;
        auto edgeSize = (this->_type1.front().rows - 1) / 2;
        for (int i = edgeSize + halfSize; i < this->_imgLikehood.rows - edgeSize - halfSize; ++i)
        {
            auto ptr = this->_imgLikehood.ptr<uchar>(i);
            for (int j = edgeSize + halfSize; j < this->_imgLikehood.cols - edgeSize - halfSize;)
            {
                // judge the max
                bool isMax = true;
                for (int k = i - halfSize; k != i + halfSize + 1; ++k)
                {
                    auto tempPtr = this->_imgLikehood.ptr<uchar>(k);
                    for (int l = j - halfSize; l != j + halfSize + 1; ++l)
                        if (tempPtr[l] > ptr[j])
                        {
                            isMax = false;
                            break;
                        }
                }
                if (isMax)
                {
                    if (ptr[j] != 0)
                        _candidates.push_back(ChessCorner{cv::Point(j, i), cv::Point2f()});
                    j += halfSize + 1;
                }
                else
                    ++j;
            }
        }
        CBDector::outputInConsole("Suppression Finished");
    }

    void CBDector::edgeOrientations()
    {
        CBDector::outputInConsole("Calculate Edge Orientations");
        // calculate the angle and weight image using the sobel method
        this->_imgAngle.create(this->_imgGray.size(), CV_32FC1);
        this->_imgWeight.create(this->_imgGray.size(), CV_32FC1);
        auto rows = this->_imgGray.rows;
        auto cols = this->_imgGray.cols;
        for (int i = 1; i != rows - 1; ++i)
        {
            auto grayPtr = this->_imgGray.ptr<uchar>(i);
            auto anglePtr = _imgAngle.ptr<_Float32>(i);
            auto weightPtr = _imgWeight.ptr<_Float32>(i);
            for (int j = 1; j != cols - 1; ++j)
            {
                auto l1 = grayPtr[j - 1 - cols];
                auto l2 = grayPtr[j - 1];
                auto l3 = grayPtr[j - 1 + cols];
                auto r1 = grayPtr[j + 1 - cols];
                auto r2 = grayPtr[j + 1];
                auto r3 = grayPtr[j + 1 + cols];
                auto m1 = grayPtr[j - cols];
                auto m2 = grayPtr[j + cols];
                // Ix
                auto Ix = -1 * l1 - 2 * l2 - 1 * l3 + r1 + 2 * r2 + r3;
                // Iy
                auto Iy = l1 + 2 * m1 + r1 - l3 - 2 * m2 - r3;
                anglePtr[j] = std::atan2(float(Iy), float(Ix));
                weightPtr[j] = std::sqrt(float(Ix) * Ix + Iy * Iy);
            }
        }
        // calculate the weighted orientation histogram
        auto ProtoSize = this->_type1.front().rows;
        auto halfSize = (ProtoSize - 1) / 2;
        for (auto iter = this->_candidates.begin(); iter != this->_candidates.end();)
        {
            auto &elem = *iter;
            auto rowIndex = elem._pos.y;
            auto colIndex = elem._pos.x;
            // [0, pi] to [0, 31]
            std::array<float, 32> binHist;
            binHist.fill(0.0);
            for (int i = rowIndex - halfSize; i != rowIndex + halfSize + 1; ++i)
            {
                auto anglePtr = _imgAngle.ptr<_Float32>(i);
                auto weightPtr = _imgWeight.ptr<_Float32>(i);
                for (int j = colIndex - halfSize; j != colIndex + halfSize + 1; ++j)
                {
                    float angle;
                    if (anglePtr[j] < 0.0)
                        angle = anglePtr[j] + M_PI;
                    else
                        angle = anglePtr[j];
                    int val = angle / M_PI * 32;
                    if (val == 32)
                        val = 0;
                    binHist.at(val) += weightPtr[j];
                }
            }
            for (int i = 1; i != 31; ++i)
                binHist.at(i) = 0.25 * binHist.at(i - 1) + 0.5 * binHist.at(i) + 0.25 * binHist.at(i + 1);
            auto p = CBDector::findModesMeanShift(binHist);
            auto index1 = p.first;
            auto index2 = p.second;
            iter->_grad = cv::Point2f(index1 / 32.0 * M_PI, index2 / 32.0 * M_PI);
            auto main = std::max(binHist.at(index1), binHist.at(index2));
            auto weightDis = std::abs(binHist.at(index1) - binHist.at(index2));
            auto binDis = std::abs(index2 - index1);
            /**
             * \brief the value of [0.66] is a threshold for erase candidates,
             *        larger value means erasing less.
             *        Change it if necessary.
             */
            if (weightDis > main * ns_test::params::edgeOrientationsThreshold)
                iter = this->_candidates.erase(iter);
            else
                ++iter;
        }
        CBDector::outputInConsole("Calculate Finished");
        return;
    }

    void CBDector::registerCorners()
    {
        this->outputInConsole("Register Corners");
        auto protoSize = this->_type1.at(0).rows;
        auto halfProtoSize = (protoSize - 1) / 2;
        for (auto &elem : this->_candidates)
        {
            auto col = elem._pos.x;
            auto row = elem._pos.y;
            auto alpha1 = std::min(elem._grad.x, elem._grad.y);
            auto alpha2 = std::max(elem._grad.x, elem._grad.y);
            auto protoTypeA = this->createCornerPrototype(alpha1, alpha2, protoSize);
            auto protoTypeB = this->createCornerPrototype(alpha1 + M_PI, alpha2 + M_PI, protoSize);
            auto protoTypeC = this->createCornerPrototype(alpha2, alpha1 + M_PI, protoSize);
            auto protoTypeD = this->createCornerPrototype(alpha2 + M_PI, alpha1, protoSize);
            // for the convolution
            int protoRowCount = 0, protoColCount = 0;
            int fTypeA = 0, fTypeB = 0, fTypeC = 0, fTypeD = 0;
            for (int k = row - halfProtoSize; k != row + halfProtoSize + 1; ++k, ++protoRowCount)
            {
                auto ptrTypeA = protoTypeA.ptr<uchar>(protoRowCount);
                auto ptrTypeB = protoTypeB.ptr<uchar>(protoRowCount);
                auto ptrTypeC = protoTypeC.ptr<uchar>(protoRowCount);
                auto ptrTypeD = protoTypeD.ptr<uchar>(protoRowCount);
                // for gray image
                auto ptrGray = this->_imgGray.ptr<uchar>(k);
                for (int l = col - halfProtoSize; l != col + halfProtoSize + 1; ++l, ++protoColCount)
                {
                    // for type 1
                    fTypeA += ptrGray[l] * ptrTypeA[protoColCount];
                    fTypeB += ptrGray[l] * ptrTypeB[protoColCount];
                    fTypeC += ptrGray[l] * ptrTypeC[protoColCount];
                    fTypeD += ptrGray[l] * ptrTypeD[protoColCount];
                }
                // rollback
                protoColCount = 0;
            }
            auto mu = 0.25 * (fTypeA + fTypeB + fTypeC + fTypeD);
            auto s1 = std::min(std::min(fTypeA, fTypeB) - mu, mu - std::min(fTypeC, fTypeD));
            auto s2 = std::min(mu - std::min(fTypeA, fTypeB), std::min(fTypeC, fTypeD) - mu);
            // calculate score
            auto score = std::max({s1, s2});
            /**
             * \brief the value of [0.0] is the threshold for the selected coeners
             *        larger value means selected less points.
             *        Change it if necessary.
             */
            if (score > ns_test::params::registerCornersThreshold)
                this->_corners.push_back(ChessCorner{elem._pos, elem._grad});
        }

        cv::Point2f gravity(0, 0);
        for (const auto &elem : this->_candidates)
        {
            gravity.x += elem._pos.x;
            gravity.y += elem._pos.y;
        }
        gravity.x /= this->_corners.size();
        gravity.y /= this->_corners.size();
        float minDis = MAXFLOAT;
        for (const auto &elem : this->_corners)
        {
            auto dis = std::pow(elem._pos.x - gravity.x, 2) + std::pow(elem._pos.y - gravity.y, 2);
            if (minDis > dis)
            {
                minDis = dis;
                this->_center = elem;
            }
        }
        this->outputInConsole("Register Finished");
        return;
    }

    void CBDector::grawChessBoard()
    {
        this->outputInConsole("Graw ChessBoard");

        auto up = this->findCornerUp(this->_center);
        auto down = this->findCornerDown(this->_center);
        auto left = this->findCornerLeft(this->_center);
        auto right = this->findCornerRight(this->_center);
        auto left_up = this->findCornerLeft(up);
        auto right_up = this->findCornerRight(up);
        auto left_down = this->findCornerLeft(down);
        auto right_down = this->findCornerRight(down);

        auto sortFun = [](const ChessCorner &c1, const ChessCorner &c2, const ChessCorner &c3)
        {
            std::vector<cv::Point> sort{c1._pos, c2._pos, c3._pos};
            std::sort(sort.begin(), sort.end(), [](const cv::Point &c1, const cv::Point &c2)
                      { return c1.x < c2.x; });
            return sort;
        };
        auto s1 = sortFun(left_up, up, right_up);
        auto s2 = sortFun(left, _center, right);
        auto s3 = sortFun(left_down, down, right_down);
        _chessboard(0, 0) = s1.at(0);
        _chessboard(0, 1) = s1.at(1);
        _chessboard(0, 2) = s1.at(2);
        _chessboard(1, 0) = s2.at(0);
        _chessboard(1, 1) = s2.at(1);
        _chessboard(1, 2) = s2.at(2);
        _chessboard(2, 0) = s3.at(0);
        _chessboard(2, 1) = s3.at(1);
        _chessboard(2, 2) = s3.at(2);
        pcl::KdTreeFLANN<pcl::PointXY>
            kdtree;
        pcl::PointCloud<pcl::PointXY>::Ptr cloud(new pcl::PointCloud<pcl::PointXY>());
        for (const auto &elem : this->_corners)
            cloud->push_back(pcl::PointXY(elem._pos.x, elem._pos.y));
        kdtree.setInputCloud(cloud);
        while (true)
        {
            ChessBoard upChess(0, 0);
            ChessBoard downChess(0, 0);
            ChessBoard leftChess(0, 0);
            ChessBoard rightChess(0, 0);
            std::vector<std::pair<float, ChessBoard *>> info;
            bool upUseful = true, downUseful = true, leftUseful = true, rightUseful = true;
            // up ---------------
            std::vector<cv::Point2f> up;
            for (int i = 0; i != this->_chessboard._cols; ++i)
            {
                auto &p3 = this->_chessboard(0, i);
                auto &p2 = this->_chessboard(1, i);
                auto &p1 = this->_chessboard(2, i);
                auto p4 = this->predictCorners(p1, p2, p3);
                std::vector<int> index;
                std::vector<float> dis;
                auto search = kdtree.nearestKSearch(p4, 1, index, dis);
                if (search > 0)
                {
                    auto p = cloud->at(index.at(0));
                    if (dis[0] > ns_test::params::grawChessBoardThreshold)
                    {
                        upUseful = false;
                        break;
                    }
                    up.push_back(cv::Point2f(p.x, p.y));
                }
                else
                {
                    up.push_back(cv::Point2f(p4.x, p4.y));
                }
            }
            if (upUseful)
            {
                upChess = this->_chessboard;
                upChess.addRowUp();
                for (int i = 0; i != upChess._cols; ++i)
                    upChess(0, i) = up.at(i);
                auto upEnergy = this->chessboard_energy(upChess);
                info.push_back(std::make_pair(upEnergy, &upChess));
            }
            // down
            std::vector<cv::Point2f> down;
            for (int i = 0; i != this->_chessboard._cols; ++i)
            {
                auto &p1 = this->_chessboard(this->_chessboard._rows - 3, i);
                auto &p2 = this->_chessboard(this->_chessboard._rows - 2, i);
                auto &p3 = this->_chessboard(this->_chessboard._rows - 1, i);
                auto p4 = this->predictCorners(p1, p2, p3);
                std::vector<int> index;
                std::vector<float> dis;
                auto search = kdtree.nearestKSearch(p4, 1, index, dis);
                if (search > 0)
                {
                    auto p = cloud->at(index.at(0));
                    if (dis[0] > ns_test::params::grawChessBoardThreshold)
                    {
                        downUseful = false;
                        break;
                    }
                    down.push_back(cv::Point2f(p.x, p.y));
                }
                else
                {
                    down.push_back(cv::Point2f(p4.x, p4.y));
                }
            }
            if (downUseful)
            {
                downChess = this->_chessboard;
                downChess.addRowDown();
                for (int i = 0; i != downChess._cols; ++i)
                    downChess(downChess._rows - 1, i) = down.at(i);
                auto downEnergy = this->chessboard_energy(downChess);
                info.push_back(std::make_pair(downEnergy, &downChess));
            }
            // left
            std::vector<cv::Point2f> left;
            for (int i = 0; i != this->_chessboard._rows; ++i)
            {
                auto &p1 = this->_chessboard(i, 2);
                auto &p2 = this->_chessboard(i, 1);
                auto &p3 = this->_chessboard(i, 0);
                auto p4 = this->predictCorners(p1, p2, p3);
                std::vector<int> index;
                std::vector<float> dis;
                auto search = kdtree.nearestKSearch(p4, 1, index, dis);
                if (search > 0)
                {
                    auto p = cloud->at(index.at(0));
                    if (dis[0] > ns_test::params::grawChessBoardThreshold)
                    {
                        leftUseful = false;
                        break;
                    }
                    left.push_back(cv::Point2f(p.x, p.y));
                }
                else
                {
                    left.push_back(cv::Point2f(p4.x, p4.y));
                }
            }
            if (leftUseful)
            {
                leftChess = this->_chessboard;
                leftChess.addColLeft();
                for (int i = 0; i != leftChess._rows; ++i)
                    leftChess(i, 0) = left.at(i);
                auto leftEnergy = this->chessboard_energy(leftChess);
                info.push_back(std::make_pair(leftEnergy, &leftChess));
            }
            // right
            std::vector<cv::Point2f> right;
            for (int i = 0; i != this->_chessboard._rows; ++i)
            {
                auto &p1 = this->_chessboard(i, this->_chessboard._cols - 3);
                auto &p2 = this->_chessboard(i, this->_chessboard._cols - 2);
                auto &p3 = this->_chessboard(i, this->_chessboard._cols - 1);
                auto p4 = this->predictCorners(p1, p2, p3);
                std::vector<int> index;
                std::vector<float> dis;
                auto search = kdtree.nearestKSearch(p4, 1, index, dis);
                if (search > 0)
                {
                    auto p = cloud->at(index.at(0));
                    if (dis[0] > ns_test::params::grawChessBoardThreshold)
                    {
                        rightUseful = false;
                        break;
                    }
                    right.push_back(cv::Point2f(p.x, p.y));
                }
                else
                {
                    right.push_back(cv::Point2f(p4.x, p4.y));
                }
            }
            if (rightUseful)
            {
                rightChess = this->_chessboard;
                rightChess.addColRight();
                for (int i = 0; i != rightChess._rows; ++i)
                    rightChess(i, rightChess._cols - 1) = right.at(i);
                auto rightEnergy = this->chessboard_energy(rightChess);
                info.push_back(std::make_pair(rightEnergy, &rightChess));
            }
            if (info.size() == 0)
                break;
            std::sort(info.begin(), info.end(), [](const std::pair<float, ChessBoard *> &p1, const std::pair<float, ChessBoard *> &p2)
                      { return p1.first < p2.first; });
            this->_chessboard = *info.front().second;
        }
        this->outputInConsole("Graw Finished");
        return;
    }

    CBDector::ChessCorner &CBDector::findCornerLeft(const ChessCorner &center)
    {
        auto gVec1 = this->gradVector(center._grad.x);
        return this->findCorner(-gVec1, center);
    }

    CBDector::ChessCorner &CBDector::findCornerRight(const ChessCorner &center)
    {
        auto gVec1 = this->gradVector(center._grad.x);
        return this->findCorner(gVec1, center);
    }

    CBDector::ChessCorner &CBDector::findCornerUp(const ChessCorner &center)
    {
        auto gVec2 = this->gradVector(center._grad.y);
        return this->findCorner(gVec2, center);
    }

    CBDector::ChessCorner &CBDector::findCornerDown(const ChessCorner &center)
    {
        auto gVec2 = this->gradVector(center._grad.y);
        return this->findCorner(-gVec2, center);
    }

    CBDector::ChessCorner &CBDector::findCorner(const cv::Vec2f &dir, const ChessCorner &p)
    {
        float minDis = MAXFLOAT;
        cv::namedWindow("win", cv::WINDOW_FREERATIO);
        cv::Mat temp0(this->_imgLikehood.size(), CV_8UC3);
        cv::drawMarker(temp0, _center._pos, cv::Scalar(0, 255, 255), 0, 40, 2);
        int count = 0;
        auto iter = this->_corners.begin(), target = this->_corners.begin();
        for (; iter != this->_corners.end(); ++iter)
        {
            auto elem = *iter;
            if (elem._pos.x == p._pos.x && elem._pos.y == p._pos.y)
                continue;
            cv::Vec2f pointTo(elem._pos.x - p._pos.x, -elem._pos.y + p._pos.y);
            auto proj = pointTo.dot(dir);
            auto norm = std::sqrt(pointTo[0] * pointTo[0] + pointTo[1] * pointTo[1]);
            if (proj < 0.0)
                norm = MAXFLOAT;
            auto dist = std::sin(std::acos(pointTo.dot(dir) / norm)) * norm;
            auto val = 5 * dist + proj;
            if (val < minDis)
            {
                minDis = val;
                target = iter;
            }
        }
        return *target;
    }

    float CBDector::chessboard_energy(const ChessBoard &chessboard)
    {
        float eMax = 0.0;
        for (int i = 0; i != chessboard._rows; ++i)
        {
            for (int j = 1; j != chessboard._cols - 1; ++j)
            {
                auto l = chessboard(i, j - 1);
                auto cur = chessboard(i, j);
                auto r = chessboard(i, j + 1);
                auto pi = Eigen::Vector2f(l.x, l.y);
                auto pj = Eigen::Vector2f(cur.x, cur.y);
                auto pk = Eigen::Vector2f(r.x, r.y);
                auto up = std::pow((pi + pk - 2 * pj).norm(), 2);
                auto down = std::pow((pi - pk).norm(), 2);
                eMax = std::max(double(eMax), up / down);
            }
        }
        for (int i = 0; i != chessboard._cols; ++i)
        {
            for (int j = 1; j != chessboard._rows - 1; ++j)
            {
                auto u = chessboard(j - 1, i);
                auto cur = chessboard(j, i);
                auto d = chessboard(j + 1, i);
                auto pi = Eigen::Vector2f(u.x, u.y);
                auto pj = Eigen::Vector2f(cur.x, cur.y);
                auto pk = Eigen::Vector2f(d.x, d.y);
                auto up = std::pow((pi + pk - 2 * pj).norm(), 2);
                auto down = std::pow((pi - pk).norm(), 2);
                eMax = std::max(double(eMax), up / down);
            }
        }
        return (eMax - 1) * chessboard._rows * chessboard._cols;
    }

    pcl::PointXY CBDector::predictCorners(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Point2f &p3)
    {
        auto x = 2 * (p3.x - p2.x) - (p2.x - p1.x) + p3.x;
        auto y = 2 * (p3.y - p2.y) - (p2.y - p1.y) + p3.y;
        return pcl::PointXY(x, y);
    }

    void CBDector::organizationMapping(float blockSize)
    {
        this->outputInConsole("Organization Mapping");

        for (int i = 0; i != this->_chessboard._rows; ++i)
            for (int j = 0; j != this->_chessboard._cols; ++j)
            {
                cv::Point2f pix = this->_chessboard(i, j);
                cv::Point2f real(j * blockSize, i * blockSize);
                this->_mapping.push_back(Mapping{pix, real});
            }
        this->outputInConsole("Finished");
        return;
    }

#pragma endregion

#pragma region public member methods

    void CBDector::process(float blockSize, std::string imgOutputDir)
    {

        bool toOutPutImgs;
        imgOutputDir == "" ? toOutPutImgs = false : toOutPutImgs = true;
        if (toOutPutImgs && imgOutputDir.back() != '/')
            imgOutputDir.push_back('/');

        this->calCornerLikehood();
        if (toOutPutImgs)
            cv::imwrite(imgOutputDir + "img_LH.png", this->_imgLikehood);

        this->nonMaximumSuppression();
        if (toOutPutImgs)
        {
            cv::Mat img_NMS(this->_imgLikehood.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            for (const auto &elem : this->_candidates)
                cv::drawMarker(img_NMS, elem._pos, cv::Scalar(0, 255, 255), 0, 40, 2);
            cv::imwrite(imgOutputDir + "img_NMS.png", img_NMS);
        }

        this->edgeOrientations();
        if (toOutPutImgs)
        {
            cv::Mat img_EO(this->_imgLikehood.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            for (const auto &elem : this->_candidates)
                cv::drawMarker(img_EO, elem._pos, cv::Scalar(0, 0, 255), 0, 40, 2);
            cv::imwrite(imgOutputDir + "img_EO.png", img_EO);
        }

        this->registerCorners();
        if (toOutPutImgs)
        {
            cv::Mat img_RC(this->_imgLikehood.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            for (const auto &elem : this->_corners)
                cv::drawMarker(img_RC, elem._pos, cv::Scalar(0, 255, 255), 0, 40, 2);
            cv::imwrite(imgOutputDir + "img_RC.png", img_RC);

            cv::Mat img_Weight(this->_imgLikehood.size(), CV_8UC1, cv::Scalar(0));
            cv::normalize(this->_imgWeight, img_Weight, 255, 0, cv::NormTypes::NORM_MINMAX, 0);
            cv::imwrite(imgOutputDir + "img_Weight.png", img_Weight);
        }

        this->grawChessBoard();

        this->organizationMapping(blockSize);
        if (toOutPutImgs)
        {
            cv::Mat img_Mapping(this->_imgLikehood.size(), CV_8UC3, cv::Scalar(0, 0, 0));
            for (const auto &elem : this->_mapping)
                cv::drawMarker(img_Mapping, elem._pixel, cv::Scalar(0, 255, 255), 0, 40, 2);
            cv::imwrite(imgOutputDir + "img_Mapping.png", img_Mapping);

            int count = 1;
            for (const auto &elem : this->_mapping)
            {
                cv::putText(this->_imgSrc, std::to_string(count), cv::Point(elem._pixel.x + 10, elem._pixel.y - 10),
                            cv::HersheyFonts::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 255), 4);
                cv::drawMarker(this->_imgSrc, elem._pixel, cv::Scalar(0, 0, 255), cv::MarkerTypes::MARKER_CROSS, 40, 5);
                ++count;
            }
            cv::imwrite(imgOutputDir + "img_Result.png", this->_imgSrc);
        }
        return;
    }

#pragma endregion
} // namespace ns_test
