#include "tflite.hpp"

int main()
{
    //const std::string mod_path = "models/图像分割/deeplabv3_1_default_1.tflite";
    //const std::string mod_path = "models/图像分割/lite-model_deeplabv3-xception65-cityscapes_1_default_1.tflite";
    const std::string mod_path = "models/图像分割/lite-model_mobilenetv2-dm05-coco_fp16_1.tflite";
    const std::string vid_path = "models/demo.mp4";
    MobilenetVideoTFliteInfer(mod_path, vid_path);
    return 0;
}


