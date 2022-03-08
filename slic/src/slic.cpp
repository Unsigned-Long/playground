#include "slic.h"
#include <set>
#include <vector>
#include <map>
#include <random>
#include <stack>

namespace ns_slic
{
    cv::Mat SLIC::process(const cv::Mat &src, std::size_t K, bool pixelize, bool drawCens, int iterCount)
    {
        // prepare
        auto N = src.rows * src.cols;
        auto S = static_cast<int>(std::sqrt(static_cast<float>(N) / K));
        auto supSize = S * S;
        auto step = S + 2;
        auto M = 10;
        auto width = src.cols;
        auto height = src.rows;
        auto rows = src.rows;
        auto cols = src.cols;

        // get lab color space image
        auto labSrc = src.clone();
        cv::cvtColor(labSrc, labSrc, cv::COLOR_BGR2Lab);

        // get the grad image
        auto grad = cv::Mat(rows, cols, CV_32FC1);
        for (int i = 1; i != rows - 1; ++i)
        {
            auto gradPtr = grad.ptr<_Float32>(i);
            auto labPtr = labSrc.ptr<uchar>(i);
            for (int j = 1; j != cols - 1; ++j)
            {
                auto lx = &(labPtr[j * 3 + 0]);
                auto ly = &(labPtr[j * 3 + 0]);
                auto ax = &(labPtr[j * 3 + 1]);
                auto ay = &(labPtr[j * 3 + 1]);
                auto bx = &(labPtr[j * 3 + 2]);
                auto by = &(labPtr[j * 3 + 2]);
                auto dx = std::pow(lx[3] - lx[-3], 2) +
                          std::pow(ax[3] - ax[-3], 2) +
                          std::pow(bx[3] - bx[-3], 2);
                auto dy = std::pow(ly[3 * cols] - ly[-3 * cols], 2) +
                          std::pow(ay[3 * cols] - ay[-3 * cols], 2) +
                          std::pow(by[3 * cols] - by[-3 * cols], 2);
                gradPtr[j] = dx + dy;
            }
        }

        // get the init centers
        std::vector<cv::Point2i> centersLastXY;
        std::vector<cv::Point3i> centersLastLab;
        for (int y = S / 2.0; y < rows; y += S)
            for (int x = S / 2.0; x < cols; x += S)
            {
                centersLastXY.push_back(cv::Point2i(x, y));
            }

        // Perturb Seeds
        for (int i = 0; i != centersLastXY.size(); ++i)
        {
            auto &cen = centersLastXY.at(i);
            auto r = cen.y;
            auto c = cen.x;
            auto curGrad = MAXFLOAT;
            int x, y, l, a, b;
            for (int i = r - 1; i != r + 1; ++i)
                for (int j = c - 1; j != c + 1; ++j)
                {
                    auto newGrad = grad.ptr<_Float32>(i)[j];
                    if (newGrad < curGrad)
                    {
                        curGrad = newGrad;
                        y = i;
                        x = j;
                        auto lab = labSrc.ptr<uchar>(i);
                        l = lab[3 * j + 0];
                        a = lab[3 * j + 1];
                        b = lab[3 * j + 2];
                    }
                }
            centersLastLab.push_back(cv::Point3i(l, a, b));
            cen.y = y;
            cen.x = x;
        }

        // iteration
        // [center index, lab]
        std::map<int, cv::Point3i> centersLab;
        std::map<int, cv::Point2i> centersXY;
        std::map<int, int> pixelCounter;

        cv::Mat label(rows, cols, CV_16UC1);

        double lastMinDis = MAXFLOAT;
        int count = 0;

        while (true)
        {
            double newMinDis = 0.0;
            // classify
            for (int i = 0; i != rows; ++i)
            {
                auto ptr = labSrc.ptr<uchar>(i);
                auto labelPtr = label.ptr<uint16_t>(i);
                for (int j = 0; j != cols; ++j)
                {
                    auto l = ptr[j * 3 + 0];
                    auto a = ptr[j * 3 + 1];
                    auto b = ptr[j * 3 + 2];
                    // find the class
                    auto minD = MAXFLOAT;
                    auto minIndex = 0;
                    for (int k = 0; k != centersLastXY.size(); ++k)
                    {
                        auto r = centersLastXY.at(k).y;
                        auto c = centersLastXY.at(k).x;
                        if (std::abs(i - r) >= S || std::abs(j - c) >= S)
                            continue;
                        auto cl = centersLastLab.at(k).x;
                        auto ca = centersLastLab.at(k).y;
                        auto cb = centersLastLab.at(k).z;
                        auto dc = std::sqrt(std::pow(l - cl, 2) + std::pow(a - ca, 2) + std::pow(b - cb, 2));
                        auto ds = std::sqrt(std::pow(r - i, 2) + std::pow(c - j, 2));
                        auto D = std::sqrt(pow(dc / M, 2) + std::pow(ds / S, 2));
                        if (D < minD)
                        {
                            minD = D;
                            minIndex = k;
                        }
                    }
                    newMinDis += minD;

                    // label for piexl
                    labelPtr[j] = minIndex;

                    centersLab[minIndex].x += l;
                    centersLab[minIndex].y += a;
                    centersLab[minIndex].z += b;

                    centersXY[minIndex].x += j;
                    centersXY[minIndex].y += i;

                    ++pixelCounter[minIndex];
                }
            }

            for (int i = 0; i != centersLastXY.size(); ++i)
            {
                auto size = pixelCounter[i];

                centersLab[i].x /= size;
                centersLab[i].y /= size;
                centersLab[i].z /= size;

                centersXY[i].x /= size;
                centersXY[i].y /= size;

                // update
                centersLastLab.at(i).x = centersLab[i].x;
                centersLastLab.at(i).y = centersLab[i].y;
                centersLastLab.at(i).z = centersLab[i].z;

                // centersLastLab.at(i).x = labSrc.ptr<uchar>(centersXY[i].y)[3 * centersXY[i].x + 0];
                // centersLastLab.at(i).y = labSrc.ptr<uchar>(centersXY[i].y)[3 * centersXY[i].x + 1];
                // centersLastLab.at(i).z = labSrc.ptr<uchar>(centersXY[i].y)[3 * centersXY[i].x + 2];

                centersLastXY.at(i).x = centersXY[i].x;
                centersLastXY.at(i).y = centersXY[i].y;

                // rollback
                pixelCounter[i] = 0;

                centersLab[i].x = 0;
                centersLab[i].y = 0;
                centersLab[i].z = 0;

                centersXY[i].x = 0;
                centersXY[i].y = 0;
            }

            auto diff = std::abs(newMinDis - lastMinDis);
            lastMinDis = newMinDis;
            ++count;

            std::cout << "diff : {" << diff << '}' << std::endl;

            if (diff < 100 || count > iterCount)
                break;
        }
        if (pixelize)
        {

            // if labeled, value equals to 1
            cv::Mat labeled(rows, cols, CV_8UC1, cv::Scalar(0));
            auto labelStartPtr = label.ptr<uint16_t>(0);
            auto labeledStartPtr = labeled.ptr<uchar>(0);
            auto labelAt = [labelStartPtr, cols](int r, int c) -> uint16_t & {
                return labelStartPtr[r * cols + c];
            };
            auto labeledAt = [labeledStartPtr, cols](int r, int c) -> uchar & {
                return labeledStartPtr[r * cols + c];
            };
            auto posCheck = [rows, cols](const cv::Point2i &pos)
            {
                return pos.x >= 0 && pos.x < cols && pos.y >= 0 && pos.y < rows;
            };

            for (int i = 0; i != rows; ++i)
                for (int j = 0; j != cols; ++j)
                {
                    auto &curLabel = labelAt(i, j);
                    auto &curLabeled = labeledAt(i, j);
                    if (curLabeled == 1)
                        continue;

                    std::stack<cv::Point2i> pixelSta;
                    pixelSta.push(cv::Point2i(j, i));

                    std::vector<cv::Point2i> regionPiexl;
                    regionPiexl.push_back(cv::Point2i(j, i));

                    std::map<int, int> boundary;

                    while (!pixelSta.empty())
                    {
                        auto seed = pixelSta.top();
                        pixelSta.pop();

                        auto left = seed + cv::Point2i(-1, 0);
                        auto up = seed + cv::Point2i(0, -1);
                        auto right = seed + cv::Point2i(1, 0);
                        auto down = seed + cv::Point2i(0, 1);
                        std::vector<cv::Point2i> prob{left, up, right, down};

                        for (const auto &elem : prob)
                            if (posCheck(elem))
                            {
                                auto elemLabel = labelAt(elem.y, elem.x);
                                if (elemLabel == curLabel && labeledAt(elem.y, elem.x) == 0)
                                {
                                    labeledAt(elem.y, elem.x) = 1;
                                    pixelSta.push(elem);
                                    regionPiexl.push_back(elem);
                                }
                                if (elemLabel != curLabel)
                                    ++boundary[elemLabel];
                            }
                    }
                    if (regionPiexl.size() < supSize / 2)
                    {
                        auto max = std::max_element(boundary.begin(), boundary.end(), [](const auto &p1, const auto &p2)
                                                    { return p1.second < p2.second; })
                                       ->first;
                        for (const auto &elem : regionPiexl)
                            labelAt(elem.y, elem.x) = max;
                    }
                }
        }

        cv::Mat dst(rows, cols, CV_8UC3);
        for (int i = 0; i != rows; ++i)
        {
            auto dstPtr = dst.ptr<uchar>(i);
            auto labelPtr = label.ptr<uint16_t>(i);
            for (int j = 0; j != cols; ++j)
            {
                auto lab = centersLastLab.at(labelPtr[j]);
                dstPtr[3 * j + 0] = lab.x;
                dstPtr[3 * j + 1] = lab.y;
                dstPtr[3 * j + 2] = lab.z;
            }
        }

        cv::cvtColor(dst, dst, cv::COLOR_Lab2BGR);

        if (drawCens)
            for (const auto &elem : centersLastXY)
                cv::circle(dst, elem, 4, cv::Scalar(0, 0, 255), 5);

        cv::medianBlur(dst, dst, 3);
        return dst;
    }
} // namespace ns_slic