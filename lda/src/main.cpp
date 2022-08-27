#include "artwork/geometry/point.hpp"
#include "lda.h"

std::vector<Eigen::Vector2d> geoPts2EigenVec(const ns_geo::PointSet2d &pts) {
  std::vector<Eigen::Vector2d> points(pts.size());
  for (int i = 0; i != pts.size(); ++i) {
    points[i](0) = pts[i].x;
    points[i](1) = pts[i].y;
  }
  return points;
}

int main(int argc, char const *argv[]) {
  auto class1 = ns_geo::PointSet2d::randomGenerator(5, 0.0, 2.0, 1.0, 4.0);
  auto class2 = ns_geo::PointSet2d::randomGenerator(5, 3.0, 4.0, 0.0, 2.0);

  class1.write("../data/class1.txt", std::ios::out);
  class2.write("../data/class2.txt", std::ios::out);

  ns_lda::Class<2> class1_t(1, geoPts2EigenVec(class1));
  ns_lda::Class<2> class2_t(2, geoPts2EigenVec(class2));

  class1_t.print();
  class2_t.print();
  Eigen::Matrix2d m;
  Eigen::EigenSolver<Eigen::Matrix2d> eigen_solver(m);
  auto eigenvalues = eigen_solver.eigenvalues();
  auto eigenvectors = eigen_solver.eigenvectors();
  Eigen::Matrix2d real = eigenvectors.real();

  auto [result, mat] = ns_lda::LDASolver::solve<2, 2, 1>({class1_t, class2_t});
  result[0].print();
  result[1].print();
  return 0;
}
