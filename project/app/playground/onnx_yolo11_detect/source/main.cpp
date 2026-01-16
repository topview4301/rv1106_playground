#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <string>

#define TIMER_START(name) auto start_##name = std::chrono::high_resolution_clock::now();
#define TIMER_END(name) \
    auto end_##name = std::chrono::high_resolution_clock::now(); \
    auto duration_##name = std::chrono::duration_cast<std::chrono::microseconds>(end_##name - start_##name).count() / 1000.0; \
    std::cout << "[" << #name << "] " << std::fixed << std::setprecision(3) << duration_##name << " ms" << std::endl;

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
    float r = std::min((float)inputWidth / image.cols, (float)inputHeight / image.rows);
    int unpadW = round(r * image.cols);
    int unpadH = round(r * image.rows);
    
    scale_x = (float)unpadW / image.cols;
    scale_y = (float)unpadH / image.rows;
    left_padding = (inputWidth - unpadW) / 2;
    top_padding = (inputHeight - unpadH) / 2;

    cv::Mat resized, p_image;
    cv::resize(image, resized, cv::Size(unpadW, unpadH));
    cv::copyMakeBorder(resized, p_image, top_padding, inputHeight - unpadH - top_padding,
                       left_padding, inputWidth - unpadW - left_padding, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat floatImage;
    p_image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

    std::vector<float> inputTensor(1 * 3 * inputHeight * inputWidth);
    std::vector<cv::Mat> channels(3);
    for (int i = 0; i < 3; ++i) {
        channels[i] = cv::Mat(inputHeight, inputWidth, CV_32FC1, &inputTensor[i * inputHeight * inputWidth]);
    }
    cv::split(floatImage, channels);
    std::swap(channels[0], channels[2]); 

    return inputTensor;
}

float calculateIoU(const cv::Rect& box1, const cv::Rect& box2) {
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    float inter = (x2 - x1) * (y2 - y1);
    return inter / (box1.width * box1.height + box2.width * box2.height - inter);
}

void nonMaxSuppression(std::vector<cv::Rect>& boxes, std::vector<float>& scores, 
                        std::vector<int>& classIds, float nmsThreshold, std::vector<int>& indices) {
    TIMER_START(NMS_Logic)
    std::vector<int> order(scores.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

    std::vector<bool> suppressed(scores.size(), false);
    for (size_t i = 0; i < order.size(); ++i) {
        int idx1 = order[i];
        if (suppressed[idx1]) continue;
        indices.push_back(idx1);
        for (size_t j = i + 1; j < order.size(); ++j) {
            int idx2 = order[j];
            if (suppressed[idx2]) continue;
            if (classIds[idx1] == classIds[idx2]) {
                if (calculateIoU(boxes[idx1], boxes[idx2]) > nmsThreshold) suppressed[idx2] = true;
            }
        }
    }
    TIMER_END(NMS_Logic)
}

void postprocess(const cv::Mat& image, const float* outputData, const std::vector<int64_t>& outputShape,
                 float confThres, float nmsThres, float sx, float sy, int lp, int tp,
                 const std::vector<std::string>& classNames, std::vector<cv::Scalar>& colors, cv::Mat& result) {
    TIMER_START(Postprocess_Total)
    result = image.clone();
    std::vector<cv::Rect> boxes;
    std::vector<float> confs;
    std::vector<int> ids;

    int dims = outputShape[1];
    int anchors = outputShape[2];

    for (int i = 0; i < anchors; ++i) {
        float maxS = 0; int id = -1;
        for (int j = 4; j < dims; ++j) {
            float s = outputData[j * anchors + i];
            if (s > maxS) { maxS = s; id = j - 4; }
        }
        if (maxS > confThres) {
            float cx = outputData[0 * anchors + i], cy = outputData[1 * anchors + i];
            float w = outputData[2 * anchors + i], h = outputData[3 * anchors + i];
            boxes.push_back(cv::Rect((cx - w/2 - lp)/sx, (cy - h/2 - tp)/sy, w/sx, h/sy));
            confs.push_back(maxS);
            ids.push_back(id);
        }
    }

    std::vector<int> indices;
    nonMaxSuppression(boxes, confs, ids, nmsThres, indices);

    for (int idx : indices) {
        cv::rectangle(result, boxes[idx], colors[ids[idx] % colors.size()], 2);
        std::string label = classNames[ids[idx]] + ": " + std::to_string((int)(confs[idx] * 100)) + "%";
        cv::putText(result, label, cv::Point(boxes[idx].x, boxes[idx].y - 5), 0, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    TIMER_END(Postprocess_Total)
}

// --- Main ---

int main(int argc, char *argv[]) {
    try {
        if (argc < 4) {
            std::cout << "Usage: ./main model.onnx labels.txt img.jpg" << std::endl;
            return 1;
        }

        std::vector<std::string> classNames;
        if (!loadClassNames(argv[2], classNames)) return 1;
        cv::Mat image = cv::imread(argv[3]);
        if (image.empty()) return 1;

        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "yolo");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(1);
        Ort::Session session(env, argv[1], options);
        Ort::AllocatorWithDefaultOptions allocator;

        auto inputDims = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        int ih = inputDims[2], iw = inputDims[3];

        TIMER_START(Preprocessing)
        float sx, sy; int lp, tp;
        std::vector<float> inputTensor = preprocess(image, iw, ih, sx, sy, lp, tp);
        TIMER_END(Preprocessing)

        std::vector<int64_t> inputShape = {1, 3, ih, iw};
        auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputVal = Ort::Value::CreateTensor<float>(mem, inputTensor.data(), inputTensor.size(), inputShape.data(), inputShape.size());
        
        auto inName = session.GetInputNameAllocated(0, allocator);
        auto outName = session.GetOutputNameAllocated(0, allocator);
        const char* inNames[] = {inName.get()};
        const char* outNames[] = {outName.get()};

        TIMER_START(Inference)
        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inNames, &inputVal, 1, outNames, 1);
        TIMER_END(Inference)

        const float* outData = outputTensors[0].GetTensorData<float>();
        auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
        auto colors = generateColors(classNames.size());
        cv::Mat result;
        postprocess(image, outData, outShape, 0.45f, 0.45f, sx, sy, lp, tp, classNames, colors, result);

        cv::imwrite("detection_result.jpg", result);
        std::cout << "Done. Result saved." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}