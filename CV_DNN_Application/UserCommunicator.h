#pragma once
#include <string>

#include "ClassifierBase.h"

namespace OpenCVApp {
	class UserCommunicator {
	public:
		static std::string askForFilename();
		static void showClassifiedResult(const cv::Mat* probatilies, const std::string& labelTxt, int showClassCount );
	};
}