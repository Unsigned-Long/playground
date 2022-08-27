#ifndef LDA_H
#define LDA_H

#define FORMAT_VECTOR
#include "artwork/logger/logger.h"
#include "eigen3/Eigen/Dense"

namespace ns_lda {

  template <int Size>
  struct Class {
    using VecType = Eigen::Vector<double, Size>;
    using MatType = Eigen::Matrix<double, Size, Size>;

  public:
    std::vector<VecType> data;
    int classId;

  public:
    Class(int classId, const std::vector<VecType> &data)
        : classId(classId), data(data) {}

    Class() = default;

    VecType meanVec() const {
      VecType center = VecType::Zero();
      for (int i = 0; i != data.size(); ++i) {
        center += data[i];
      }
      center /= data.size();
      return center;
    }

    MatType getSwMat() const {
      MatType Sw = MatType::Zero();
      VecType mean = meanVec();
      for (int i = 0; i != data.size(); ++i) {
        const auto &item = data[i];
        Sw += (item - mean) * (item - mean).transpose();
      }
      return Sw;
    }

    void print() {
      LOG_VAR(classId);
      std::cout << std::fixed << std::setprecision(20) << std::endl;
      for (const auto &elem : data) {
        std::cout << elem.transpose() << '\n';
      }
    }
  };

  class LDASolver {
  public:
    template <int SrcSize, int Classes, int DstSize>
    static std::pair<std::array<Class<DstSize>, Classes>, Eigen::Matrix<double, DstSize, SrcSize>>
    solve(const std::array<Class<SrcSize>, Classes> &data) {
      if (DstSize <= 0 || DstSize > Classes) {
        throw std::runtime_error("wrong dst size");
      }

      using VecType = Eigen::Vector<double, SrcSize>;
      using MatType = Eigen::Matrix<double, SrcSize, SrcSize>;

      MatType Sw = MatType::Zero();
      MatType Sb = MatType::Zero();
      MatType St = MatType::Zero();
      VecType mean_t = VecType::Zero();

      // Sw
      for (int i = 0; i != Classes; ++i) {
        Sw += data[i].getSwMat();
        mean_t += data[i].meanVec();
      }

      // total mean
      mean_t /= Classes;

      // St
      for (int i = 0; i != Classes; ++i) {
        for (const auto &item : data[i].data) {
          St += (item - mean_t) * (item - mean_t).transpose();
        }
      }

      // Sb
      Sb = St - Sw;

      MatType SwInv;
      {
        // SVD decomposition
        Eigen::JacobiSVD<MatType> svd(Sw, Eigen::ComputeFullU | Eigen::ComputeFullV);
        auto singularVal = svd.singularValues();
        MatType sigmaMatrix = MatType::Zero();
        for (int i = 0; i != SrcSize; ++i) {
          sigmaMatrix(i, i) = singularVal(i);
        }
        auto uMatrix = svd.matrixU();
        auto vMatrix = svd.matrixV();
        SwInv = vMatrix * sigmaMatrix.inverse() * uMatrix.transpose();
      }

      MatType mat = SwInv * Sb;
      Eigen::EigenSolver<MatType> eigen_solver(mat);
      auto eigenvalues = eigen_solver.eigenvalues();
      auto eigenvectors = eigen_solver.eigenvectors().real();

      auto wMat = eigenvectors.block(0, 0, SrcSize, DstSize);
      Eigen::Matrix<double, DstSize, SrcSize> wMatTrans = wMat.transpose();

      std::array<Class<DstSize>, Classes> result;

      for (int i = 0; i != result.size(); ++i) {
        auto &curClass = result[i];
        curClass.classId = data[i].classId;
        curClass.data.resize(data[i].data.size());
        for (int j = 0; j != data[i].data.size(); ++j) {
          curClass.data[j] = wMatTrans * data[i].data[j];
        }
      }

      return {result, wMatTrans};
    }
  };
} // namespace ns_lda

#endif