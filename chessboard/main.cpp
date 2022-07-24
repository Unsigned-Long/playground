#include "artwork/flags/flags.hpp"
#include "artwork/logger/logger.h"
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

int main(int argc, char const *argv[]) {
  try {
    FLAGS_DEF_INT(size, s, "the size of each block", ns_flags::OptionProp::OPTIONAL, 100);
    FLAGS_DEF_INT(rows, r, "the row count of the chess board", ns_flags::OptionProp::OPTIONAL, 5);
    FLAGS_DEF_INT(cols, c, "the cols count of the chess board", ns_flags::OptionProp::OPTIONAL, 8);

    FLAGS_DEF_NO_OPTION(STRING, imgSavePath, "the save path of the chess board image",
                        ns_flags::OptionProp::OPTIONAL, "../images/board.png");

    ns_flags::setupFlags(argc, argv);

    if (flags_size <= 0) {
      throw std::runtime_error("the 'size' is invalid");
    }
    if (flags_rows <= 0) {
      throw std::runtime_error("the 'rows' is invalid");
    }
    if (flags_cols <= 0) {
      throw std::runtime_error("the 'cols' is invalid");
    }

    LOG_VAR(flags_size, flags_imgSavePath);
    LOG_VAR(flags_rows, flags_cols);

    flags_cols += 1;
    flags_rows += 1;


    cv::Mat board(flags_rows * flags_size, flags_cols * flags_size, CV_8UC1, cv::Scalar(255));

    for (int i = 0; i != flags_rows; ++i) {
      for (int j = 0; j != flags_cols; ++j) {
        if ((i + j) % 2 == 0) {
          board(cv::Rect2i(j * flags_size, i * flags_size, flags_size, flags_size)).setTo(0);
        }
      }
    }

    cv::imwrite(flags_imgSavePath, board);

  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
  }

  return 0;
}
