#pragma unmanaged
#include "ClassifierBase.h"

#include<iostream>
#include<fstream>
#include <math.h>

namespace OpenCVApp {

	ClassifierBase::ClassifierBase()
	{
		/* do nothing */
	}

	ClassifierBase::~ClassifierBase()
	{
		/* do nothing */
	}
	
	void ClassifierBase::outLayerAsImage(const std::string& layer, const std::string& nameHead, const std::string& extension, int number) {
		cv::Mat blob = net.forward(layer);
		int channels = blob.size[1];
		int rows = blob.size[2];
		int cols = blob.size[3];
		char numstr[5] = {};

		int imageSize = rows * cols;
		float* headPtr = (float*)blob.ptr<float>() + channels * imageSize * number;

		for (int i = 0; i < channels; i++) {
			cv::Mat out = cv::Mat(rows, cols, CV_32F,
				headPtr + imageSize * i);

			sprintf_s(numstr, "%04d", i);
			cv::imwrite(nameHead + std::string(numstr) + extension, out);
		}
	}
	void ClassifierBase::outLayerAsCsv(const std::string& layer, const std::string& nameHead, int number ) {
		cv::Mat blob = net.forward(layer);
		int channels = blob.size[1];
		int rows = blob.size[2];
		int cols = blob.size[3];
		char numstr[5] = {};

		int imageSize = rows * cols;

		std::ofstream csv(nameHead + ".csv");
		float* headPtr = (float*)blob.ptr<float>() + channels * imageSize * number;

		for (int c = 0; c < channels; c++) {
			csv << "channel" << c << "\n";
			for (int x = 0; x < rows; x++) {
				for (int y = 0; y < rows; y++) {
					csv << headPtr[c * imageSize + x * rows + y] << ",";
				}
				csv << "\n";
			}
		}
		csv.close();
	}

	void ClassifierBase::getMaxClass(const cv::Mat* probabilities, int *topClass, double *topProb) {
		cv::Point classNumber;
		minMaxLoc(*probabilities, NULL, topProb, NULL, &classNumber);
		*topClass = classNumber.x;
	}
	void ClassifierBase::getRankHigherClasses(const cv::Mat* probabilities, std::vector<int> *higherClass, std::vector<float> *higherProbs, int rankCount) {
		cv::Mat higherClassIndexs;
		cv::sortIdx(*probabilities, higherClassIndexs, CV_SORT_EVERY_ROW | CV_SORT_DESCENDING);

		for (int i = 0; i < std::min(rankCount, higherClassIndexs.cols); i++) {
			int classIndex = higherClassIndexs.at<int>(0, i);
			higherClass->push_back(classIndex);
			higherProbs->push_back(probabilities->at<float>(0, classIndex));
		}

	}

	std::vector<std::string> ClassifierBase::readClassNames(const std::string& filename) {
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