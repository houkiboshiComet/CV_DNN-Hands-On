#pragma unmanaged
#include "UserCommunicator.h"

#include <stdlib.h>
#include <windows.h>
#include <math.h>

namespace OpenCVApp {
	
	std::string UserCommunicator::askForFilename() {
		OPENFILENAMEA ofn;
		char nameBuff[MAX_PATH] = "";
		
		ZeroMemory(&ofn, sizeof(ofn));
		ofn.lStructSize = sizeof(OPENFILENAME);
		ofn.lpstrFilter = "‰æ‘œƒtƒ@ƒCƒ‹(*.jpg,*.jpeg,,*.bmp,*.png)\0*.jpg;*.bmp;*.png;*.jpeg\0";
		ofn.lpstrFile = nameBuff;
		ofn.nMaxFile = MAX_PATH;
		ofn.Flags = OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
		if (GetOpenFileNameA(&ofn)) {
			return std::string(nameBuff);
		}
		else {
			return "";
		}
	}

	void UserCommunicator::showClassifiedResult(const cv::Mat* probatilies, const std::string& labelTxt, int showClassCount) {
			
		std::vector<int> rankHigerClassIds;
		std::vector<float> rankHigerClassProbs;

		ClassifierBase::getRankHigherClasses(probatilies, &rankHigerClassIds, &rankHigerClassProbs, showClassCount);

		std::vector<std::string> classNames = ClassifierBase::readClassNames(labelTxt);

		showClassCount = min(showClassCount, (int) rankHigerClassIds.size());

		std::cout << showClassCount << " Best Classified Classes" << std::endl;

		for (int i = 0; i < showClassCount; i++) {
			int classId = rankHigerClassIds[i];
			std::cout << "\t" << "Top " << i + 1 << " Class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
			std::cout << "\t" << "Probability: " << rankHigerClassProbs[i] * 100 << "%" << std::endl;
		}
	}
	
}