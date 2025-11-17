#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "ffi/torch_ffi.h"
#include "python_hook.h"
#include <string>
#include <map>
#include <iostream>
#include <iomanip>

using namespace tvm_sdk::ffi;
using namespace tvm_sdk;
using ::testing::Not;
using ::testing::IsEmpty;
using ::testing::HasSubstr;

// Test fixture for TorchFFI tests
class TorchFFITest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Python
        PythonHook::initialize();
    }

    void TearDown() override {
        // Cleanup if needed
    }

    void print_map(const std::map<std::string, std::string>& map_data, const std::string& prefix = "  ") {
        for (const auto& pair : map_data) {
            std::cout << prefix << std::setw(25) << std::left << pair.first << ": " << pair.second << "\n";
        }
    }
};

// Test: Load ResNet18 model with pretrained weights
TEST_F(TorchFFITest, LoadResNet18Pretrained) {
    std::cout << "\n[Test] Loading ResNet18 with pretrained weights...\n";

    auto model_info = TorchFFI::load_resnet18(true);

    EXPECT_THAT(model_info, Not(IsEmpty()));
    EXPECT_EQ(model_info["pretrained"], "true");
    EXPECT_EQ(model_info["status"], "loaded");
    EXPECT_EQ(model_info["model_name"], "ResNet18");

    std::cout << "Model Info:\n";
    print_map(model_info);
}

// Test: Load ResNet18 model without pretrained weights
TEST_F(TorchFFITest, LoadResNet18NoPretrained) {
    std::cout << "\n[Test] Loading ResNet18 without pretrained weights...\n";

    auto model_info = TorchFFI::load_resnet18(false);

    EXPECT_THAT(model_info, Not(IsEmpty()));
    EXPECT_EQ(model_info["pretrained"], "false");
    EXPECT_EQ(model_info["status"], "loaded");
}

// Test: Get model information
TEST_F(TorchFFITest, GetModelInfo) {
    std::cout << "\n[Test] Getting ResNet18 model info...\n";

    auto model_info = TorchFFI::get_model_info();

    EXPECT_THAT(model_info, Not(IsEmpty()));
    EXPECT_EQ(model_info["model_name"], "ResNet18");
    EXPECT_EQ(model_info["num_classes"], "1000");
    EXPECT_THAT(model_info["total_params"], Not(IsEmpty()));

    std::cout << "Model Architecture:\n";
    print_map(model_info);
}

// Test: Get traced model info
TEST_F(TorchFFITest, GetTracedModelInfo) {
    std::cout << "\n[Test] Getting TorchScript traced model info...\n";

    auto traced_info = TorchFFI::get_traced_model_info(1, 224, 224);

    EXPECT_EQ(traced_info["status"], "traced");
    EXPECT_EQ(traced_info["format"], "torchscript");
    EXPECT_EQ(traced_info["method"], "trace");

    std::cout << "Traced Model Info:\n";
    print_map(traced_info);
}

// Test: Save model state
TEST_F(TorchFFITest, SaveModelState) {
    std::cout << "\n[Test] Saving ResNet18 model state...\n";

    std::string output_path = "test_resnet18_state.pth";
    auto save_info = TorchFFI::save_model_state(output_path);

    EXPECT_EQ(save_info["status"], "success");
    EXPECT_EQ(save_info["output_path"], output_path);
    EXPECT_EQ(save_info["format"], "state_dict");

    std::cout << "Save Info:\n";
    print_map(save_info);
}

// Test: Predict image with dog.jpeg
TEST_F(TorchFFITest, PredictDogImage) {
    std::cout << "\n[Test] Running inference on dog.jpeg...\n";

    // Path to dog.jpeg using project root
    std::string project_root = PROJECT_SOURCE_DIR;
    std::string image_path = project_root + "/test/dog.jpeg";

    try {
        // Call Python predict_image function
        py::object result = PythonHook::call_function(
            "torch_ext",
            "predict_image",
            image_path
        );

        py::dict dict_result = py::cast<py::dict>(result);

        // Extract status
        std::string status = py::cast<std::string>(dict_result["status"]);
        EXPECT_EQ(status, "success");

        if (status == "success") {
            // Extract top-1 prediction
            std::string top1_class = py::cast<std::string>(dict_result["top1_class"]);
            double top1_prob = py::cast<double>(dict_result["top1_probability"]);

            // Extract inference timing info
            std::string avg_time = py::cast<std::string>(dict_result["avg_inference_time_ms"]);
            std::string std_time = py::cast<std::string>(dict_result["std_inference_time_ms"]);
            std::string min_time = py::cast<std::string>(dict_result["min_inference_time_ms"]);
            std::string max_time = py::cast<std::string>(dict_result["max_inference_time_ms"]);
            std::string num_iterations = py::cast<std::string>(dict_result["num_iterations"]);

            std::cout << "\n=== Inference Results ===\n";
            std::cout << "Image: " << image_path << "\n";
            std::cout << "Top-1 Class: " << top1_class << "\n";
            std::cout << "Top-1 Probability: " << std::fixed << std::setprecision(4) << (top1_prob * 100) << "%\n";

            std::cout << "\n=== Inference Time (PyTorch) ===\n";
            std::cout << "Iterations: " << num_iterations << "\n";
            std::cout << "Average: " << avg_time << " ms\n";
            std::cout << "Std Dev: " << std_time << " ms\n";
            std::cout << "Min: " << min_time << " ms\n";
            std::cout << "Max: " << max_time << " ms\n";

            // Extract top-5 predictions
            py::list top5_predictions = py::cast<py::list>(dict_result["top5_predictions"]);

            std::cout << "\n=== Top-5 Predictions ===\n";
            std::cout << std::string(60, '-') << "\n";
            std::cout << std::setw(5) << "Rank"
                      << std::setw(10) << "Class ID"
                      << std::setw(30) << "Class Name"
                      << std::setw(15) << "Probability\n";
            std::cout << std::string(60, '-') << "\n";

            for (size_t i = 0; i < py::len(top5_predictions); i++) {
                py::dict pred = py::cast<py::dict>(top5_predictions[i]);
                int class_id = py::cast<int>(pred["class_id"]);
                std::string class_name = py::cast<std::string>(pred["class_name"]);
                double probability = py::cast<double>(pred["probability"]);

                std::cout << std::setw(5) << (i + 1)
                          << std::setw(10) << class_id
                          << std::setw(30) << class_name
                          << std::setw(14) << std::fixed << std::setprecision(2) << (probability * 100) << "%\n";
            }
            std::cout << std::string(60, '-') << "\n";

            // Verify top-1 class contains reasonable prediction
            // (Should be some kind of dog breed)
            EXPECT_THAT(top1_class, Not(IsEmpty()));
            EXPECT_GT(top1_prob, 0.0);
            EXPECT_LE(top1_prob, 1.0);

        } else {
            std::string error_msg = py::cast<std::string>(dict_result["error_message"]);
            FAIL() << "Prediction failed: " << error_msg;
        }

    } catch (const std::exception& e) {
        FAIL() << "Exception during prediction: " << e.what();
    }
}

// Test: Get ImageNet classes
TEST_F(TorchFFITest, GetImageNetClasses) {
    std::cout << "\n[Test] Getting ImageNet class labels...\n";

    try {
        py::object result = PythonHook::call_function(
            "torch_ext",
            "get_imagenet_classes"
        );

        py::dict dict_result = py::cast<py::dict>(result);
        int num_classes = py::cast<int>(dict_result["num_classes"]);

        EXPECT_EQ(num_classes, 1000);

        std::cout << "ImageNet classes: " << num_classes << "\n";

        // Print sample classes
        py::dict sample_classes = py::cast<py::dict>(dict_result["sample_classes"]);
        std::cout << "\nSample classes (first 10):\n";
        for (auto item : sample_classes) {
            std::string key = py::cast<std::string>(item.first);
            std::string value = py::cast<std::string>(item.second);
            std::cout << "  Class " << key << ": " << value << "\n";
        }

    } catch (const std::exception& e) {
        FAIL() << "Exception: " << e.what();
    }
}

// Main function
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
