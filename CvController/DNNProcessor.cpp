#include "ImageProcessor.h"

#pragma unmanaged
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>


#include "DNNProcessor.h"

using cv::Mat;
using namespace cv::dnn::experimental_dnn_v2;

namespace OpenCVApp {
	
	DNNProcessor::DNNProcessor()
	{
		net = new cv::dnn::experimental_dnn_v2::Net();
	}

	DNNProcessor::~DNNProcessor()
	{
		delete net;
	}

	GoogLeNetProcessor::GoogLeNetProcessor() 
	{
	}
	GoogLeNetProcessor::~GoogLeNetProcessor()
	{
	}
	
	void GoogLeNetProcessor::createNeuralNet() {
		static const std::string model = "data\\GoogLeNet\\bvlc_googlenet.caffemodel";
		static const std::string prototxt = "data\\GoogLeNet\\bvlc_googlenet.prototxt";
		Net net = cv::dnn::experimental_dnn_v2::readNetFromCaffe(model, prototxt);
	}
	void GoogLeNetProcessor::applyNeuralNet(const Mat* image, void* output) {
		int cropSize = 224;
		Mat smallImage;
		cv::resize(*image, smallImage, cv::Size(cropSize, cropSize));
		cv::Mat blob = cv::dnn::experimental_dnn_v2::blobFromImage(*image);
		
		net->setInput(blob);
		Mat result = net->forward();
	}
}