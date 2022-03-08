#include "artwork/flags/flags.hpp"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

using namespace ns_flags;

int main(int argc, char const *argv[]) {
  try {
    OptionParser parser;
    parser.addOption<IntArg>("size", 100, "the width of each block");
    parser.addOption<IntArg>("rows", 7, "the row count of the chess board");
    parser.addOption<IntArg>("cols", 10, "the cols count of the chess board");

    parser.setDefaultOption<StringArg>("../images/board-100-7-10.png");

    parser.setupParser(argc, argv);

    int size = parser.getOptionArgv<IntArg>("size");
    int rows = parser.getOptionArgv<IntArg>("rows");
    int cols = parser.getOptionArgv<IntArg>("cols");
    std::string path = parser.getDefaultOptionArgv<StringArg>();

    if (size <= 0) {
      throw std::runtime_error("the 'size' is invalid");
    }
    if (rows <= 0) {
      throw std::runtime_error("the 'rows' is invalid");
    }
    if (cols <= 0) {
      throw std::runtime_error("the 'cols' is invalid");
    }

    cv::Mat board(rows * size, cols * size, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i != rows; ++i) {
      for (int j = 0; j != cols; ++j) {
        if ((i + j) % 2 == 0) {
          board(cv::Rect2i(j * size, i * size, size, size)).setTo(0);
        }
      }
    }

    cv::imwrite(path, board);

  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }

  return 0;
}
