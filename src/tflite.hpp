#pragma once
#ifndef TFLITE_DEMO
#define TFLITE_DEMO

#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/model.h"


void MobilenetVideoTFliteInfer(const std::string& modelfile, const std::string& videofile);

#endif
