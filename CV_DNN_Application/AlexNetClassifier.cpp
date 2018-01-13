#pragma unmanaged
#include "AlexNetClassifier.h"


namespace OpenCVApp {
	
	AlexNetClassifier::AlexNetClassifier() {

	}
	AlexNetClassifier::~AlexNetClassifier() {

	}
	void AlexNetClassifier::createNeuralNet() {
		static const std::string model = "..\\model\\AlexNet\\bvlc_alexnet.caffemodel";
		static const std::string prototxt = "..\\model\\AlexNet\\deploy.prototxt";
		net = cv::dnn::readNetFromCaffe(prototxt, model);
	}

	void AlexNetClassifier::applyNeuralNet(const cv::Mat* image, cv::Mat* output) {
		static const cv::Size cropSize(227, 227);
		static const cv::Scalar averageColor(104, 117, 123);

		cv::Mat blob = cv::dnn::blobFromImage(*image, 1, cropSize, averageColor);
		net.setInput(blob);

		//中間層が確認できることのサンプルコード ハンズオンで確認するのも面白いのでは。
		cv::Mat conv = net.forward("pool1");
		for (int i = 0; i < conv.size[1]; i++) {
			cv::Mat out = cv::Mat(conv.size[2], conv.size[3], CV_32F, conv.ptr<float>() + 
				conv.size[2] * conv.size[3] * i);
			cv::imwrite("out" + std::to_string( i ) + ".bmp", out);
		}
		
		*output = net.forward();
	}
}