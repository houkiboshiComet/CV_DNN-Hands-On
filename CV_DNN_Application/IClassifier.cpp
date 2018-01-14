#pragma unmanaged
#include "IClassifier.h"

namespace OpenCVApp {
	
	IClassifier::IClassifier()
	{
		/* do nothing */
	}

	IClassifier::~IClassifier()
	{
		/* do nothing */
	}


	void IClassifier::outLayerAsImage(const std::string& layer, const std::string& nameHead, const std::string& extension, int number) {
		cv::Mat blob = net.forward(layer);
		int channels = blob.size[1];
		int rows = blob.size[2];
		int cols = blob.size[3];
		char numstr[5] = {};

		int imageSize = rows * cols;
		float* headPtr = (float*) blob.ptr<float>() + channels * imageSize * number;

		for (int i = 0; i < channels; i++) {
			cv::Mat out = cv::Mat(rows, cols, CV_32F,
				headPtr + imageSize * i);

			sprintf_s(numstr, "%04d", i);
			cv::imwrite(nameHead + std::string(numstr) + extension, out);
		}
	}

	void IClassifier::getMaxClass(const cv::Mat* probabilities, int *classId, double *classProb) {
		cv::Mat probMat = probabilities->reshape(1, 1); //reshape the probabilities to 1x1000 matrix
		cv::Point classNumber;
		minMaxLoc(probMat, NULL, classProb, NULL, &classNumber);
		*classId = classNumber.x;
	}
	std::vector<std::string> IClassifier::readClassNames(const std::string& filename) {
		std::vector<std::string> classNames;
		std::ifstream fp(filename);
		if (!fp.is_open())
		{
			throw std::ios_base::failure("File with classes labels not found: " + filename);
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