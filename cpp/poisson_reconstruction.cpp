#include <open3d/Open3D.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Options {
    std::string input_path;
    std::string output_path;
    double voxel_size = 0.01;
    double normal_radius = 0.05;
    int normal_max_nn = 50;
    int orient_max_nn = 100;
    int poisson_depth = 7;
    double scale = 1.1;
    double density_quantile = 0.02;
    bool linear_fit = false;
};

void PrintProgress(int step, int total, const std::string& message) {
    constexpr int width = 30;
    const double ratio = total > 0 ? static_cast<double>(step) / total : 1.0;
    const int filled = static_cast<int>(ratio * width);
    std::cout << "["
              << std::string(filled, '#')
              << std::string(width - filled, '-')
              << "] " << static_cast<int>(ratio * 100.0)
              << "%  " << message << std::endl;
}

double ParseDouble(const std::string& value, const std::string& name) {
    try {
        return std::stod(value);
    } catch (...) {
        throw std::runtime_error("Invalid numeric value for " + name + ": " + value);
    }
}

int ParseInt(const std::string& value, const std::string& name) {
    try {
        return std::stoi(value);
    } catch (...) {
        throw std::runtime_error("Invalid integer value for " + name + ": " + value);
    }
}

Options ParseArgs(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: poisson_reconstruction_cpp <input_point_cloud> <output_mesh> "
            "[--voxel-size 0.01] [--normal-radius 0.05] [--normal-max-nn 50] "
            "[--orient-max-nn 100] [--poisson-depth 7] [--scale 1.1] "
            "[--density-quantile 0.02] [--linear-fit]");
    }

    Options options;
    options.input_path = argv[1];
    options.output_path = argv[2];

    for (int i = 3; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_value = [&](const std::string& name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("Missing value for " + name);
            }
            return argv[++i];
        };

        if (arg == "--voxel-size") {
            options.voxel_size = ParseDouble(require_value(arg), arg);
        } else if (arg == "--normal-radius") {
            options.normal_radius = ParseDouble(require_value(arg), arg);
        } else if (arg == "--normal-max-nn") {
            options.normal_max_nn = ParseInt(require_value(arg), arg);
        } else if (arg == "--orient-max-nn") {
            options.orient_max_nn = ParseInt(require_value(arg), arg);
        } else if (arg == "--poisson-depth") {
            options.poisson_depth = ParseInt(require_value(arg), arg);
        } else if (arg == "--scale") {
            options.scale = ParseDouble(require_value(arg), arg);
        } else if (arg == "--density-quantile") {
            options.density_quantile = ParseDouble(require_value(arg), arg);
        } else if (arg == "--linear-fit") {
            options.linear_fit = true;
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (options.normal_radius <= 0) throw std::runtime_error("--normal-radius must be positive.");
    if (options.normal_max_nn < 3) throw std::runtime_error("--normal-max-nn must be at least 3.");
    if (options.orient_max_nn < 3) throw std::runtime_error("--orient-max-nn must be at least 3.");
    if (options.poisson_depth < 4) throw std::runtime_error("--poisson-depth must be at least 4.");
    if (options.density_quantile < 0 || options.density_quantile >= 1) {
        throw std::runtime_error("--density-quantile must be in [0, 1).");
    }

    return options;
}

double Quantile(std::vector<double> values, double q) {
    if (values.empty()) {
        return 0.0;
    }
    const double position = q * static_cast<double>(values.size() - 1);
    const auto index = static_cast<size_t>(position);
    std::nth_element(values.begin(), values.begin() + index, values.end());
    return values[index];
}

int main(int argc, char** argv) {
    try {
        const Options options = ParseArgs(argc, argv);
        const int total_steps = 6;

        PrintProgress(1, total_steps, "读取点云");
        auto point_cloud = std::make_shared<open3d::geometry::PointCloud>();
        if (!open3d::io::ReadPointCloud(options.input_path, *point_cloud) || point_cloud->IsEmpty()) {
            throw std::runtime_error("输入点云为空或读取失败: " + options.input_path);
        }
        std::cout << "读取点云: " << options.input_path
                  << " 点数=" << point_cloud->points_.size() << std::endl;

        PrintProgress(2, total_steps, "体素下采样");
        if (options.voxel_size > 0.0) {
            point_cloud = point_cloud->VoxelDownSample(options.voxel_size);
        }
        std::cout << "下采样后点数=" << point_cloud->points_.size() << std::endl;

        PrintProgress(3, total_steps, "估计点云法线");
        point_cloud->EstimateNormals(
            open3d::geometry::KDTreeSearchParamHybrid(options.normal_radius, options.normal_max_nn));

        PrintProgress(4, total_steps, "统一法线方向");
        point_cloud->OrientNormalsConsistentTangentPlane(options.orient_max_nn);

        PrintProgress(5, total_steps, "Poisson 重建");
        std::shared_ptr<open3d::geometry::TriangleMesh> mesh;
        std::vector<double> densities;
        std::tie(mesh, densities) = open3d::geometry::TriangleMesh::CreateFromPointCloudPoisson(
            *point_cloud,
            options.poisson_depth,
            0.0f,
            static_cast<float>(options.scale),
            options.linear_fit);

        std::cout << "泊松重建完成: 顶点数=" << mesh->vertices_.size()
                  << " 三角形数=" << mesh->triangles_.size() << std::endl;

        if (options.density_quantile > 0.0 && !densities.empty()) {
            const double threshold = Quantile(densities, options.density_quantile);
            std::vector<bool> remove_mask(densities.size(), false);
            for (size_t i = 0; i < densities.size(); ++i) {
                remove_mask[i] = densities[i] < threshold;
            }
            mesh->RemoveVerticesByMask(remove_mask);
        }

        mesh = mesh->Crop(point_cloud->GetAxisAlignedBoundingBox());
        mesh->RemoveDegenerateTriangles();
        mesh->RemoveDuplicatedTriangles();
        mesh->RemoveDuplicatedVertices();
        mesh->RemoveNonManifoldEdges();
        mesh->ComputeVertexNormals();

        PrintProgress(6, total_steps, "保存网格");
        const fs::path output_path(options.output_path);
        if (output_path.has_parent_path()) {
            fs::create_directories(output_path.parent_path());
        }
        if (!open3d::io::WriteTriangleMesh(options.output_path, *mesh)) {
            throw std::runtime_error("保存网格失败: " + options.output_path);
        }

        std::cout << "保存重建网格: " << options.output_path
                  << " 顶点数=" << mesh->vertices_.size()
                  << " 三角形数=" << mesh->triangles_.size() << std::endl;
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "Error: " << exc.what() << std::endl;
        return 1;
    }
}
