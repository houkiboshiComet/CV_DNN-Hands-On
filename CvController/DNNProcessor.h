#pragma once

#include <string>
namespace cv{
	class Mat;
	namespace dnn {
		namespace experimental_dnn_v2 {
			class Net;
		}
		//class Blob;
	}
}
using cv::dnn::experimental_dnn_v2::Net;
using cv::Mat;

namespace OpenCVApp {

	class  DNNProcessor
	{
	public:
		DNNProcessor();
		virtual ~DNNProcessor();
		virtual void createNeuralNet() = 0;
		virtual void applyNeuralNet(const Mat* image, void* output ) = 0;

	protected:
		cv::dnn::Net* net;
	};

	class GoogLeNetProcessor : DNNProcessor {
	public:
		GoogLeNetProcessor();
		~GoogLeNetProcessor();
		void createNeuralNet();
		void applyNeuralNet(const Mat* image, void* output);
	};

}