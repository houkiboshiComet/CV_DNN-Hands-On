// CV_DNN_Application.cpp : アプリケーションのエントリ ポイントを定義します。
//
#include "AlexNetClassifier.h"
#include "GoogLeNetClassifier.h"

#include <fstream>
#include <iostream>
#include <cstdlib>
using namespace std;

using namespace OpenCVApp;


int main(int argc, char **argv)
{
	int ret = 0;
	IClassifier* dnnClassifier = new AlexNetClassifier();

	try {
		cv::Mat image = cv::imread("..\\image\\cat.jpg", -1);
		cv::Mat result;

		dnnClassifier->createNeuralNet();
		dnnClassifier->applyNeuralNet(&image, &result);

		int classId;
		double classProb;
		
		IClassifier::getMaxClass(&result, &classId, &classProb);//find the best class
		std::vector<std::string> classNames = IClassifier::readClassNames("..\\model\\synset_words.txt");
		std::cout << "Best class: #" << classId << " '" << classNames.at(classId) << "'" << std::endl;
		std::cout << "Probability: " << classProb * 100 << "%" << std::endl;
	}
	catch (cv::Exception& e) {
		std::cout << e.what() << std::endl;
		ret = 1;
	}
	catch (std::exception& e) {
		std::cout << e.what() << std::endl;
		ret = 1;
	}
	
	delete dnnClassifier;
	return ret;
}
