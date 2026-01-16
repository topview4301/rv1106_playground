#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <iomanip>

#define TIMER_START(name) auto start_##name = std::chrono::high_resolution_clock::now();
#define TIMER_END(name, task_name) \
    auto end_##name = std::chrono::high_resolution_clock::now(); \
    auto duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(end_##name - start_##name).count() / 1000.0; \
    std::cout << "[TIMER] " << task_name << ": " << std::fixed << std::setprecision(2) << duration_##name << " ms" << std::endl;

std::vector<cv::Scalar> generateColors(int numClasses) {
    std::vector<cv::Scalar> colors;
    srand(42); 
    for (int i = 0; i < numClasses; i++) {
        colors.push_back(cv::Scalar(rand() % 256, rand() % 256, rand() % 256));
    }
    return colors;
}

bool loadClassNames(const std::string& configPath, std::vector<std::string>& classNames) {
    std::ifstream file(configPath);
    if (!file.is_open()) return false;
    classNames.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (!line.empty()) classNames.push_back(line);
    }
    return !classNames.empty();
}

std::vector<float> preprocess(const cv::Mat& image, int inputWidth, int inputHeight,
                             float& scale_x, float& scale_y, int& left_padding, int& top_padding) {
    cv::Mat processedImage;
    float r = std::min((float)inputWidth / (float)image.cols, (float)inputHeight / (float)image.rows);
    int newUnpadWidth = round(r * image.cols);
    int newUnpadHeight = round(r * image.rows);

    scale_x = (float)newUnpadWidth / (float)image.cols;
    scale_y = (float)newUnpadHeight / (float)image.rows;
    left_padding = (inputWidth - newUnpadWidth) / 2;
    top_padding = (inputHeight - newUnpadHeight) / 2;

    cv::resize(image, processedImage, cv::Size(newUnpadWidth, newUnpadHeight));
    cv::copyMakeBorder(processedImage, processedImage, top_padding, inputHeight - newUnpadHeight - top_padding,
                       left_padding, inputWidth - newUnpadWidth - left_padding, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat floatImage;
    processedImage.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    std::vector<float> inputTensor(1 * 3 * inputHeight * inputWidth);
    for (int c = 0; c < 3; c++) {
        for (int h = 0; h < inputHeight; h++) {
            for (int w = 0; w < inputWidth; w++) {
                inputTensor[c * inputHeight * inputWidth + h * inputWidth + w] = floatImage.at<cv::Vec3f>(h, w)[2 - c];
            }
        }
    }
    return inputTensor;
}

void postprocess(const cv::Mat& image, const float* outputData, const std::vector<int64_t>& outputShape,
                 float confidenceThreshold, float scale_x, float scale_y, int left_padding, int top_padding,
                 const std::vector<std::string>& classNames, std::vector<cv::Scalar>& colors, cv::Mat& resultImage) {
    
    resultImage = image.clone();
    int numDetections = (int)outputShape[1];
    int elementsPerRow = (int)outputShape[2];

    for (int i = 0; i < numDetections; ++i) {
        const float* row = outputData + (i * elementsPerRow);
        float score = row[4];

        if (score > confidenceThreshold) {
            float x1 = row[0], y1 = row[1], x2 = row[2], y2 = row[3];
            int classId = (int)row[5];

            float origX1 = (x1 - left_padding) / scale_x;
            float origY1 = (y1 - top_padding) / scale_y;
            float origX2 = (x2 - left_padding) / scale_x;
            float origY2 = (y2 - top_padding) / scale_y;

            cv::Rect box(cv::Point(std::max(0.0f, origX1), std::max(0.0f, origY1)),
                         cv::Point(std::min((float)image.cols, origX2), std::min((float)image.rows, origY2)));

            cv::Scalar color = colors[classId % colors.size()];
            cv::rectangle(resultImage, box, color, 2);
            std::string label = (classId < classNames.size() ? classNames[classId] : "Unknown") + " " + std::to_string((int)(score * 100)) + "%";
            int baseLine;
            cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::rectangle(resultImage, cv::Point(box.x, box.y - labelSize.height - 10), cv::Point(box.x + labelSize.width, box.y), color, cv::FILLED);
            cv::putText(resultImage, label, cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        if (argc < 4) {
            std::cerr << "Usage: " << argv[0] << " <model.onnx> <label.txt> <image.jpg> [conf]" << std::endl;
            return 1;
        }

        std::string modelPath = argv[1], labelPath = argv[2], imagePath = argv[3];
        float confThreshold = (argc > 4) ? std::stof(argv[4]) : 0.25f;

        std::vector<std::string> classNames;
        if (!loadClassNames(labelPath, classNames)) return 1;

        cv::Mat image = cv::imread(imagePath);
        if (image.empty()) return 1;

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLO26");
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;

        auto inputName = session.GetInputNameAllocated(0, allocator);
        auto inputShape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (inputShape[0] == -1) inputShape[0] = 1;
        if (inputShape[2] <= 0) inputShape[2] = 640;
        if (inputShape[3] <= 0) inputShape[3] = 640;

        float sx, sy; int lp, tp;
        TIMER_START(preprocess)
        std::vector<float> inputData = preprocess(image, (int)inputShape[3], (int)inputShape[2], sx, sy, lp, tp);
        TIMER_END(preprocess, "Preprocessing  ")

        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputData.data(), inputData.size(), inputShape.data(), inputShape.size());

        auto outputName = session.GetOutputNameAllocated(0, allocator);
        const char* inputNames[] = { inputName.get() };
        const char* outputNames[] = { outputName.get() };

        TIMER_START(inference)
        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        TIMER_END(inference, "Inference      ")

        const float* rawOutput = outputTensors[0].GetTensorData<float>();
        auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

        std::vector<cv::Scalar> colors = generateColors((int)classNames.size());
        cv::Mat result;
        TIMER_START(postprocess)
        postprocess(image, rawOutput, outShape, confThreshold, sx, sy, lp, tp, classNames, colors, result);
        TIMER_END(postprocess, "Post-processing")

        cv::imwrite("yolo26_output.jpg", result);
        std::cout << "[SUCCESS] Result saved to yolo26_output.jpg" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }
    return 0;
}