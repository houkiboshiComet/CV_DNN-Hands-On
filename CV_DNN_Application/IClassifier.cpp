#pragma unmanaged
#include "IClassifier.h"

namespace OpenCVApp {
	
	IClassifier::IClassifier()
	{
	}

	IClassifier::~IClassifier()
	{
	}

	void IClassifier::getMaxClass(const cv::Mat* probabilities, int *classId, double *classProb) {
		cv::Mat probMat = probabilities->reshape(1, 1); //reshape the probabilities to 1x1000 matrix
		cv::Point classNumber;
		minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
		*classId = classNumber.x;
	}
	std::vector<std::string> IClassifier::readClassNames(const char *filename) {
		std::vector<std::string> classNames;
		std::ifstream fp(filename);
		if (!fp.is_open())
		{
			throw std::ios_base::failure("File with classes labels not found: " + std::string(filename));
		}
		std::string name;
		while (!fp.eof())
		{
			std::getline(fp, name);
			if (name.length() > 0) {
				classNames.push_back(name.substr(name.find(' ') + 1));
			}
		}
		fp.close();
		return classNames;
	}
}