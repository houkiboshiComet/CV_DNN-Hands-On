#pragma unmanaged
#include "ClassifierBase.h"

#include<iostream>
#include<fstream>
#include <math.h>
#include "Paths.h"

namespace OpenCVApp {

	ClassifierBase::ClassifierBase()
	{
		/* do nothing */
	}

	ClassifierBase::~ClassifierBase()
	{
		/* do nothing */
	}

	void ClassifierBase::outLayerAsImage(const std::string& layer) {
		const std::string outDir = "..\\layers\\";
		Paths::createDirAsNecessary(outDir);

		outLayerAsImage(layer, outDir + layer + "_", ".png");
	}
	void ClassifierBase::outLayerAsCsv(const std::string& layer) {
		const std::string outDir = "..\\layers\\";
		Paths::createDirAsNecessary(outDir);

		outLayerAsCsv(layer, outDir + layer);
	}
	void ClassifierBase::outLayerAsImage(const std::string& layer, const std::string& nameHead, const std::string& extension) {
		cv::Mat blob = net.forward(layer);
		int channels = 1, rows = 1, cols = 1;
		if (blob.dims >= 1) {
			cols = blob.size[blob.dims - 1];
		}
		if (blob.dims >= 2) {
			rows = blob.size[blob.dims - 2];
		}
		if (blob.dims >= 3) {
			channels = blob.size[blob.dims - 3];
		}

		char numstr[5] = {};

		int imageSize = rows * cols;
		float* headPtr = (float*)blob.ptr<float>() + channels;

		for (int i = 0; i < channels; i++) {
			cv::Mat out = cv::Mat(rows, cols, CV_32F,
				headPtr + imageSize * i);

			sprintf_s(numstr, "%04d", i);
			cv::imwrite(nameHead + std::string(numstr) + extension, out);
		}
	}
	void ClassifierBase::outLayerAsCsv(const std::string& layer, const std::string& nameHead) {
		cv::Mat blob = net.forward(layer);
		int channels = 1, rows = 1, cols = 1;
		if (blob.dims >= 1) {
			cols = blob.size[blob.dims - 1];
		}
		if (blob.dims >= 2) {
			rows = blob.size[blob.dims - 2];
		}
		if (blob.dims >= 3) {
			channels = blob.size[blob.dims - 3];
		}

		int imageSize = rows * cols;

		std::ofstream csv(nameHead + ".csv");
		float* headPtr = (float*)blob.ptr<float>() + channels;

		for (int c = 0; c < channels; c++) {
			csv << "channel" << c << "\n";
			for (int y = 0; y < rows; y++) {
				for (int x = 0; x < cols; x++) {
					csv << headPtr[c * imageSize + y * cols + x] << ",";
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