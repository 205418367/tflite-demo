#include "tflite.hpp"


struct Object {
	cv::Rect rec;
	int      class_id;
	float    prob;
};

float expit(float x) {
	return 1.f / (1.f + expf(-x));
}

//nms
float iou(cv::Rect& rectA, cv::Rect& rectB)
{
	int x1 = std::max(rectA.x, rectB.x);
	int y1 = std::max(rectA.y, rectB.y);
	int x2 = std::min(rectA.x + rectA.width, rectB.x + rectB.width);
	int y2 = std::min(rectA.y + rectA.height, rectB.y + rectB.height);
	int w = std::max(0, (x2 - x1 + 1));
	int h = std::max(0, (y2 - y1 + 1));
	float inter = w * h;
	float areaA = rectA.width * rectA.height;
	float areaB = rectB.width * rectB.height;
	float o = inter / (areaA + areaB - inter);
	return (o >= 0) ? o : 0;
}

void nms(std::vector<Object>& boxes, const double nms_threshold)
{
	std::vector<int> scores;
	for (int i = 0; i < boxes.size(); i++) {
		scores.push_back(boxes[i].prob);
	}
	std::vector<int> index;
	for (int i = 0; i < scores.size(); ++i) {
		index.push_back(i);
	}
	sort(index.begin(), index.end(), [&](int a, int b) {
		return scores[a] > scores[b]; });
	std::vector<bool> del(scores.size(), false);
	for (size_t i = 0; i < index.size(); i++) {
		if (!del[index[i]]) {
			for (size_t j = i + 1; j < index.size(); j++) {
				if (iou(boxes[index[i]].rec, boxes[index[j]].rec) > nms_threshold) {
					del[index[j]] = true;
				}
			}
		}
	}
	std::vector<Object> new_obj;
	for (const auto i : index) {
		Object obj;
		if (!del[i])
		{
			obj.class_id = boxes[i].class_id;
			obj.rec.x = boxes[i].rec.x;
			obj.rec.y = boxes[i].rec.y;
			obj.rec.width = boxes[i].rec.width;
			obj.rec.height = boxes[i].rec.height;
			obj.prob = boxes[i].prob;
		}
		new_obj.push_back(obj);
	}
	boxes = new_obj;
}


void MobilenetVideoTFliteInfer(const std::string& modelfile, const std::string& videofile) {
	// Load model
	std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(modelfile.c_str());
	// Build the interpreter
	tflite::ops::builtin::BuiltinOpResolver resolver;
	std::unique_ptr<tflite::Interpreter> interpreter;
	tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
	
	TfLiteTensor* output_locations = nullptr;
	TfLiteTensor* output_classes = nullptr;
	TfLiteTensor* num_detections = nullptr;
	TfLiteTensor* output_scores = nullptr;

	//auto cam = cv::VideoCapture(0);
	auto cam = cv::VideoCapture(videofile);

	//std::vector<std::string> labels;
	//std::ifstream input(labelfile);
	//for (std::string line; getline(input, line); )
	//{
	//	labels.push_back(line);
	//}

	auto cam_width = cam.get(CV_CAP_PROP_FRAME_WIDTH);
	auto cam_height = cam.get(CV_CAP_PROP_FRAME_HEIGHT);

	cv::Mat image0, image, resimage;
	for (;;) {
		cam >> image0;
		resize(image0, resimage, cv::Size(192, 192), 0, 0, cv::INTER_LINEAR);
                cv::cvtColor(resimage, image, cv::COLOR_BGR2RGB);

		interpreter->AllocateTensors();
		// float、uchar是输入的图片类型，取决于模型
		float* input = interpreter->typed_input_tensor<float>(0);
		memcpy(input, image.data, image.total() * image.elemSize());

		interpreter->SetAllowFp16PrecisionForFp32(true);
		interpreter->SetNumThreads(2);
		interpreter->Invoke();

/*
                //目标检测
		output_locations = interpreter->tensor(interpreter->outputs()[0]);
		float* output_data = output_locations->data.f;
		output_classes = interpreter->tensor(interpreter->outputs()[1]);
		float* out_cls = output_classes->data.f;
		output_scores = interpreter->tensor(interpreter->outputs()[2]);
		float* out_score = output_scores->data.f;
		num_detections = interpreter->tensor(interpreter->outputs()[3]);
		float* nums = num_detections->data.f;
*/

		std::cout<<"########################################"<<std::endl;
		output_locations = interpreter->tensor(interpreter->outputs()[0]);
		float* output_data = output_locations->data.f;
		//output_scores = interpreter->tensor(interpreter->outputs()[1]);
		//float* scores = output_scores->data.f;
                //std::cout<<"scores "<<*scores<<std::endl;
		std::cout<<"########################################"<<std::endl;

                return;
	}
}

/*
                int count = 0;
		std::vector<Object> objects;
		std::vector<float> locations;
		std::vector<float> cls;
		for (int i = 0; i < 20; i++) {
			auto output = output_data[i];
			locations.push_back(output);
			cls.push_back(out_cls[i]);
		}
		

		for (int j = 0; j < locations.size(); j += 4) {
			auto ymin = locations[j] * cam_height;
			auto xmin = locations[j + 1] * cam_width;
			auto ymax = locations[j + 2] * cam_height;
			auto xmax = locations[j + 3] * cam_width;
			auto width = xmax - xmin;
			auto height = ymax - ymin;
			// auto rec = Rect(xmin, ymin, width, height);
			float score = expit(out_score[count]);
			// std::cout << "score: "<< score << std::endl;
			// if (score < 0.5f) continue;
			// auto id=outputClasses;
			Object object;
			object.class_id = cls[count];
			object.rec.x = xmin;
			object.rec.y = ymin;
			object.rec.width = width;
			object.rec.height = height;
			object.prob = score;
			objects.push_back(object);
			count += 1;
		}
		nms(objects, 0.5);
		cv::RNG rng(12345);
		for (int l = 0; l < objects.size(); l++)
		{
			Object object = objects.at(l);
			auto score = object.prob;
			if (score < 0.60f) continue;
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			auto cls = object.class_id;
			cv::rectangle(image0, object.rec, color, 1);
			cv::putText(image0, "test: "+ std::to_string(score), cv::Point(object.rec.x, object.rec.y - 5),
				cv::FONT_HERSHEY_COMPLEX, .8, cv::Scalar(10, 255, 30));
		}
		cv::imshow("cam", image0);
		if (cv::waitKey(30) >= 0) break;
	}
	cv::destroyAllWindows();
	return;
}
*/
