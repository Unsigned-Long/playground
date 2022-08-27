#include "lda.h"

std::vector<std::string> __split__(const std::string &str, char splitor, bool ignoreEmpty = true) {
  std::vector<std::string> vec;
  auto iter = str.cbegin();
  while (true) {
    auto pos = std::find(iter, str.cend(), splitor);
    auto elem = std::string(iter, pos);
    if (!(elem.empty() && ignoreEmpty)) {
      vec.push_back(elem);
    }
    if (pos == str.cend()) {
      break;
    }
    iter = ++pos;
  }
  return vec;
}

int main(int argc, char const *argv[]) {
  std::fstream file("../data/data.csv", std::ios::in);
  std::string strLine;
  std::getline(file, strLine);
  ns_lda::Class<176> q1(1, {}), q2(2, {}), q3(3, {}), q4(4, {});
  while (std::getline(file, strLine)) {
    auto strVec = __split__(strLine, ',');
    Eigen::Vector<double, 176> vec;
    for (int i = 1; i != 177; ++i) {
      vec[i - 1] = std::stod(strVec[i]);
    }
    auto flag = strVec[0];
    if (flag == "Q1") {
      q1.data.push_back(vec);
    } else if (flag == "Q2") {
      q2.data.push_back(vec);
    } else if (flag == "Q3") {
      q3.data.push_back(vec);
    } else if (flag == "Q4") {
      q4.data.push_back(vec);
    } else {
      throw std::runtime_error("");
    }
  }
  file.close();

  auto [result, mat] = ns_lda::LDASolver::solve<176, 4, 2>({q1, q2, q3, q4});
  std::fstream file2("../data/data_after.csv", std::ios::out);
  file2 << std::fixed << std::setprecision(20);
  for (int i = 0; i != result.size(); ++i) {
    for (int j = 0; j != result[i].data.size(); ++j) {
      file2 << 'Q' << result[i].classId << ','
            << result[i].data[j](0) << ','
            << result[i].data[j](1) << '\n';
    }
  }
  file2.close();
  return 0;
}
