# 点云三维重建方法集合

本仓库用来实现点云重建，包含多种点云三维重建方法。每种方法对应一个独立的 Python 脚本，所有脚本都可以从统一配置文件 `reconstruction_config.json` 读取输入点云路径、输出文件名和算法参数。

项目约定：以后只要代码、配置、命令、参数、输出结构或依赖发生变化，都要同步更新本文档。

代码阅读说明：`poisson_reconstruction.py` 已补充较详细中文注释，重点解释 Poisson 重建流程、去噪、补点、法线估计、密度裁剪和网格清理的作用。

## 文件说明

- `superquadric_reconstruction.py`：单体 Superquadric / Superellipsoid 参数化拟合。
- `poisson_reconstruction.py`：Poisson Surface Reconstruction，适合重建兔子这类复杂表面。
- `ball_pivoting_reconstruction.py`：Ball Pivoting 表面重建。
- `alpha_shape_reconstruction.py`：Alpha Shape 表面重建。
- `convex_hull_reconstruction.py`：Convex Hull 凸包重建。
- `marching_cubes_reconstruction.py`：基于点云隐式场的 Marching Cubes 重建。
- `train_alpha_parameters.py`：Alpha Shape 参数训练/寻优脚本，用多份点云寻找更合适的默认 `alpha`。
- `train_poisson_parameters.py`：Poisson 参数训练/寻优脚本，用多份点云寻找更合适的默认 Poisson 参数组合。
- `train_poisson_compare.py`：同时运行粗到细搜索和 Optuna 搜索，分别输出两套 Poisson 参数配置，方便人工对比两个重建结果。
- `view_pcd.py`：PCD/点云显示脚本，用 Open3D 打开点云窗口。
- `generate_bolt_model.py`：生成可在 Blender 中查看、可用于 3D 打印检查的螺栓和配套螺母 OBJ/STL 模型。
- `scene_point_cloud_completion.py`：面向 Livox/教室这类大场景点云的几何补全脚本，使用局部平面 MLS 风格增密，不依赖单物体补全网络。
- `streamin/file_stream_websocket.py`：WebSocket 文件流 1 接收端，按二进制分片接收点云/任意文件并保存到 `input/`。
- `streamin/reconstruct_forward_websocket.py`：WebSocket 重建转发服务，接收文件流后执行 Poisson 三维重建，再把输出网格转发到下一个端口。
- `streamout/send_file_websocket.py`：WebSocket 文件流 1 传输端，用于本机测试文件分片传输。
- `reconstruction_common.py`：公共工具文件，包含通用命令行参数、配置读取、点云读取、输出路径组织、Open3D 网格写入、JSON 写入、法线估计等功能。
- `cpp/poisson_reconstruction.cpp`：Poisson 重建的 C++ 版本入口。
- `cpp/CMakeLists.txt`：C++ 版本的 CMake 构建文件。
- `reconstruction_config.json`：所有方法共用的输入路径和参数配置。
- `requirements.txt`：Python 依赖列表。

## 安装依赖

创建并激活虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

本项目统一使用 `.venv`。如果终端前缀显示 `(.venv-1)`，先执行 `deactivate`，再运行：

```powershell
.\activate_project_env.ps1
```

后续命令都使用：

```powershell
.\.venv\Scripts\python.exe
```

如果访问 PyPI 不稳定，可以使用清华镜像：

```powershell
$env:PIP_NO_INDEX=$null
python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn -r requirements.txt
```

当前依赖：

- `numpy`
- `scipy`
- `open3d`
- `scikit-image`
- `optuna`
- `websockets`

## WebSocket 文件流 1

启动接收服务：

```powershell
.\.venv\Scripts\python.exe streamin\file_stream_websocket.py --host 127.0.0.1 --port 8765 --save-dir input
```

服务只接收 `stream_id=1`。客户端协议：

1. 发送文本 JSON：`{"type":"start","stream_id":1,"filename":"cloud.ply","size":12345}`
2. 连续发送二进制分片，每个 WebSocket binary message 是文件的一段 bytes。
3. 发送文本 JSON：`{"type":"end","stream_id":1}`

服务会返回 `ack`、`progress`、`done` 或 `error` JSON。收到的文件默认写入 `input/`，如果同名文件已存在，会自动追加序号避免覆盖。

本机发送测试：

```powershell
.\.venv\Scripts\python.exe streamout\send_file_websocket.py input\object.xyz --url ws://127.0.0.1:8765
```

### 接收、重建、转发流水线

这个模式包含两个端口：

- `8765`：接收输入点云文件流，接收完成后执行 Poisson 三维重建。
- 下一个端口：由外部接收节点提供，用来接收重建后的输出网格文件流。

启动接收、重建、转发服务，并把 `--forward-url` 指向对方写好的接收节点：

```powershell
.\.venv\Scripts\python.exe streamin\reconstruct_forward_websocket.py --port 8765 --forward-url ws://127.0.0.1:8766
```

然后从传输端发送输入点云到 `8765`：

```powershell
.\.venv\Scripts\python.exe streamout\send_file_websocket.py input\book_pitch30_500k_1779020224_970493008.pcd --url ws://127.0.0.1:8765
```

流水线会把输入文件保存到 `input/stream_pipeline/`，把 Poisson 重建结果保存到 `model_outputs/stream_pipeline/`，并把生成的 `.ply` 网格继续发送到 `--forward-url`。

`streamin/file_stream_websocket.py` 仍然保留为本地调试用的简易接收端；正式对接时不用启动它。

如果暂时没有真实点云输入，想先测试普通文件传输，可以使用 `--passthrough` 跳过三维重建。例如发送一个 `helloworld` 文本文件：

```powershell
.\.venv\Scripts\python.exe streamin\reconstruct_forward_websocket.py --host 0.0.0.0 --port 8765 --forward-url ws://输出端IP:输出端端口 --passthrough
```

另一个终端发送测试文件：

```powershell
.\.venv\Scripts\python.exe streamout\send_file_websocket.py streamout\helloworld.txt --url ws://127.0.0.1:8765
```

也可以直接持续发送 `model_outputs\book_pitch30_500k_poisson\ply\book_pitch30_500k_poisson.ply` 到对方输出节点：

```powershell
.\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --url ws://192.168.1.131:8765 --interval 1
```

这个脚本默认使用 `raw-file` 模式：先发送一条 `file_timestamp` JSON 元信息，然后直接把 `.ply` 文件二进制分片发给对方。时间戳字段包括 `sent_at_unix_ms`、`sent_at_unix_ns` 和 `sent_at_perf_ns`。要改发 `.obj` 或 `.json`，传入 `--file`：

```powershell
.\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --url ws://192.168.1.131:8765 --file model_outputs\book_pitch30_500k_poisson\obj\book_pitch30_500k_poisson.obj --interval 1
```

如果对方只能接收纯 `.ply` bytes，不要任何元信息，使用 `--no-timestamp`：

```powershell
.\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --url ws://192.168.1.131:8765 --interval 1 --no-timestamp
```

如果只想继续测试纯文本 `helloworld`，使用 `--protocol text`：

```powershell
.\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --url ws://192.168.1.131:8765 --interval 1 --protocol text
```

如果对方接收端要求本项目之前的 `start -> binary -> end` 文件流协议，使用 `--protocol file-stream`：

```powershell
.\.venv\Scripts\python.exe streamout\send_helloworld_loop.py --url ws://192.168.1.131:8765 --interval 1 --protocol file-stream
```

## 统一配置文件

`reconstruction_config.json` 保存每种方法的输入点云地址和具体参数。例如：

```json
{
  "methods": {
    "poisson": {
      "input": "object.xyz",
      "output": "object_poisson.obj",
      "depth": 8
    }
  }
}
```

从配置文件运行所有方法：

```powershell
.\.venv\Scripts\python.exe superquadric_reconstruction.py --config reconstruction_config.json
.\.venv\Scripts\python.exe poisson_reconstruction.py --config reconstruction_config.json
.\.venv\Scripts\python.exe ball_pivoting_reconstruction.py --config reconstruction_config.json
.\.venv\Scripts\python.exe alpha_shape_reconstruction.py --config reconstruction_config.json
.\.venv\Scripts\python.exe convex_hull_reconstruction.py --config reconstruction_config.json
.\.venv\Scripts\python.exe marching_cubes_reconstruction.py --config reconstruction_config.json
```

命令行参数会覆盖配置文件中的默认值，例如：

```powershell
.\.venv\Scripts\python.exe poisson_reconstruction.py --config reconstruction_config.json --depth 9
```

通用命令行参数由 `reconstruction_common.py` 统一定义，包括：

- `--config`
- `--config-key`
- `--input`
- `--output`
- `--params-out`
- `--output-root`
- `--flat-output`

每个方法脚本只定义自己的算法参数。

## main.py 快速运行

`main.py` 是一个直接运行 Poisson 重建的简化入口，输入点云路径和输出网格路径作为两个位置参数：

```powershell
.\.venv\Scripts\python.exe main.py cloud_registered_20260511_114914.pcd model_outputs\main\main_poisson.ply
```

`main.py` 和 `poisson_reconstruction.py` 会在读取点云、估计法线、统一法线方向、Poisson 重建、保存网格等阶段显示文本进度条。

如果使用绝对路径：

```powershell
.\.venv\Scripts\python.exe C:\Users\36451\Desktop\Open3D\main.py C:\Users\36451\Desktop\Open3D\cloud_registered_20260511_114914.pcd C:\Users\36451\Desktop\Open3D\model_outputs\main\main_poisson.ply
```

## C++ Poisson 版本

项目中新增了 C++ 版本入口：

```text
cpp/poisson_reconstruction.cpp
```

注意：Python 版 Open3D 的核心算法本身已经由 C++ 实现，所以 C++ 版本主要减少 Python 层调度开销；Poisson 重建本身的速度提升不一定非常大。要编译 C++ 版本，需要安装或编译带 CMake 配置的 Open3D C++ 开发包，单独的 Python `open3d` wheel 通常不等于 C++ 开发环境。

构建示例：

```powershell
cmake -S cpp -B cpp\build -DOpen3D_DIR="C:\path\to\Open3D\lib\cmake\Open3D"
cmake --build cpp\build --config Release
```

运行示例：

```powershell
.\cpp\build\Release\poisson_reconstruction_cpp.exe input\livox_raw_120deg_500k_20260511_121137.pcd model_outputs\main\main_poisson_cpp.ply --voxel-size 0.01 --normal-radius 0.05 --normal-max-nn 50 --orient-max-nn 100 --poisson-depth 7 --scale 1.1 --density-quantile 0.05
```

C++ 版本参数含义与 `main.py` 基本一致：

- `--voxel-size`：体素下采样大小。
- `--normal-radius`：估计法线的搜索半径。
- `--normal-max-nn`：估计法线最多使用的邻居数量。
- `--orient-max-nn`：统一法线方向时参考的邻居数量。
- `--poisson-depth`：Poisson 八叉树深度。
- `--scale`：Poisson 重建尺度。
- `--density-quantile`：删除低密度顶点的分位数。
- `--linear-fit`：启用 Poisson 的线性拟合选项。

## 输出目录结构

默认输出会按照模型名和格式自动分文件夹保存：

```text
model_outputs/
  object_poisson/
    obj/
      object_poisson.obj
    ply/
      object_poisson.ply
    json/
      poisson_params.json
```

如果想使用旧的平铺输出方式，可以添加：

```powershell
--flat-output
```

## 读取输出模型

Windows 路径不要直接写成 `"model_outputs\object_alpha_shape\ply\object_alpha_shape.ply"`，因为反斜杠可能被 Python 当成转义字符。推荐使用 `pathlib.Path` 拼接路径：

```python
import open3d as o3d
from pathlib import Path

mesh_path = Path("model_outputs") / "object_alpha_shape" / "ply" / "object_alpha_shape.ply"
mesh = o3d.io.read_triangle_mesh(str(mesh_path))

if mesh.is_empty():
    raise RuntimeError(f"读取模型失败或模型为空: {mesh_path}")

mesh.compute_vertex_normals()
print(mesh)
print(f"vertices: {len(mesh.vertices)}")
print(f"triangles: {len(mesh.triangles)}")
```

如果读取的是点云文件，例如 `object.xyz`，可以使用：

```python
import open3d as o3d

pcd = o3d.io.read_point_cloud("object.xyz")
print(pcd)
```

## 显示 PCD 点云

使用 `view_pcd.py` 可以直接打开 `.pcd` 点云窗口。默认读取 `cloud_registered_20260511_114914.pcd`：

```powershell
.\.venv\Scripts\python.exe view_pcd.py
```

指定其他点云：

```powershell
.\.venv\Scripts\python.exe view_pcd.py cloud_registered_20260511_114914.pcd
```

显示前下采样并调整点大小：

```powershell
.\.venv\Scripts\python.exe view_pcd.py cloud_registered_20260511_114914.pcd --voxel-size 0.005 --point-size 3
```

查看 `.npy` 深度图反投影后的点云：

```powershell
.\.venv\Scripts\python.exe view_pcd.py input\depth_20260510_125751.npy --voxel-size 0.02 --point-size 2
```

## 支持的输入格式

- `.xyz`、`.txt`、`.csv`：读取前三个数值列作为 XYZ 坐标。
- `.npy`：支持 `(N, 3)` / `(H, W, 3)` 的 XYZ 数组，也支持二维深度图。二维深度图会用默认针孔相机模型反投影成点云；如果深度最大值大于 `100`，默认按毫米除以 `1000` 转成米。
- `.ply`、`.pcd`：通过 Open3D 读取。

## 方法说明

### Superquadric

配置运行：

```powershell
.\.venv\Scripts\python.exe superquadric_reconstruction.py --config reconstruction_config.json
```

直接运行：

```powershell
.\.venv\Scripts\python.exe superquadric_reconstruction.py --input object.xyz --output object_superquadric.obj --params-out params.json --resolution 64
```

核心公式：

```text
G = [ ( |x/a1|^(2/e2) + |y/a2|^(2/e2) )^(e2/e1) + |z/a3|^(2/e1) ]^(e1/2)
surface: G = 1
residual = (G - 1) * mean(a1, a2, a3)
```

主要参数：

- `resolution`：生成网格的采样分辨率。
- `f_scale`：`soft_l1` 鲁棒损失的尺度参数。
- `max_nfev`：非线性最小二乘优化的最大迭代评估次数。

适用场景：

适合用少量可解释参数近似一个光滑、紧凑、单体的形状，例如椭球、圆角盒、胶囊体。它不适合表达兔子的耳朵、腿、凹陷和复杂拓扑。

### Poisson Surface Reconstruction

配置运行：

```powershell
.\.venv\Scripts\python.exe poisson_reconstruction.py --config reconstruction_config.json
```

当前 `reconstruction_config.json` 中 Poisson 的默认输入为：

```text
input/livox_raw_120deg_500k_20260511_121137.pcd
```

运行时会像 `main.py` 一样打印读取点云、泊松重建完成、保存重建网格和保存参数的位置，例如：

```text
读取点云: input/livox_raw_120deg_500k_20260511_121137.pcd 点数=...
泊松重建完成: 顶点数=... 三角形数=...
保存重建网格: model_outputs\object_poisson\obj\object_poisson.obj 顶点数=... 三角形数=...
保存重建网格: model_outputs\object_poisson\ply\object_poisson.ply 顶点数=... 三角形数=...
保存参数: model_outputs\object_poisson\json\poisson_params.json
```

直接运行：

```powershell
.\.venv\Scripts\python.exe poisson_reconstruction.py --input object.xyz --output object_poisson.obj --params-out poisson_params.json --depth 8 --normals-radius 0.02
```

核心公式：

```text
给定有向法线场 V，求解指示函数 chi：
Delta chi = divergence(V)
surface = { x | chi(x) = iso_value }
```

Poisson 方法会估计一个实体指示函数，使其梯度尽量匹配点云法线场，然后提取等值面。

如果输入 `.pcd/.ply` 点云带有颜色，Poisson 重建后会把每个网格顶点赋值为最近原始点的颜色。建议查看和保存彩色结果时优先使用 `.ply`，因为 OBJ 对顶点颜色的支持不稳定，Open3D 也会提示 `Write OBJ can not include triangle normals` 一类警告。

主要参数：

- `depth`：八叉树深度。越大细节越多，但内存和时间开销越高。
- `scale`：Open3D Poisson 重建的尺度参数。
- `voxel_size`：Poisson 重建前的体素下采样大小。点云很大时必须适当增大，例如 500000 点可以先用 `0.05` 或 `0.1`。
- `linear_fit`：是否使用线性插值。
- `normals_radius`：估计法线时的邻域半径。
- `normals_max_nn`：估计法线时最多使用的邻居数量。
- `orient_max_nn`：统一法线方向时参考的邻居数量。它和 `normals_max_nn` 可以不同，前者控制方向传播一致性，后者控制局部平面拟合。
- `density_quantile`：删除低密度顶点的分位数，用于减少漂浮碎片。
- `statistical_outlier_nb_neighbors`：统计离群点去除的邻居数量，`0` 表示关闭。
- `statistical_outlier_std_ratio`：统计离群点阈值，越小删除越严格。
- `radius_outlier_nb_points`：半径离群点去除要求的最少邻居数，`0` 表示关闭。
- `radius_outlier_radius`：半径离群点去除的搜索半径。
- `completion_rounds`：局部点云补全/增密轮数，`0` 表示关闭。
- `completion_max_distance`：补全时允许连接的最大近邻距离，`0` 表示自动估计阈值。

当前 `reconstruction_config.json` 中 Poisson 使用常规下采样，并开启一轮轻量点云补全：

```text
voxel_size = 0.05
completion_rounds = 1
```

这表示先做 `0.05` 的体素下采样，再在相近点之间插入中点做一轮局部补点。它比直接使用 50 万原始点更快，也比完全关闭补点更容易让 Poisson 形成连续表面。

Poisson 重建前的预处理顺序为：

```text
读取点云 -> 体素下采样 -> 噪点处理 -> 点云补全 -> 法线估计/统一 -> Poisson 重建
```

噪点处理包含两种方式：

```text
统计离群点去除：删除平均邻居距离明显异常的点。
半径离群点去除：删除指定半径内邻居数量不足的孤立点。
```

点云补全目前是保守的局部增密：在距离较近的相邻点之间插入中点，用来缓解采样稀疏和小裂缝。它不是语义补全，不能凭空恢复被遮挡的桌椅、墙面或物体背面。点云本身缺失很大时，仍然建议多视角采集或分区域建模。

带噪点处理和一轮补全的运行示例：

```powershell
.\.venv\Scripts\python.exe poisson_reconstruction.py --input input\livox_raw_120deg_500k_20260511_121137.pcd --output livox_poisson_clean_complete.obj --params-out livox_poisson_clean_complete.json --voxel-size 0.05 --depth 7 --normals-radius 0.05 --normals-max-nn 50 --orient-max-nn 100 --density-quantile 0.05 --statistical-outlier-nb-neighbors 20 --statistical-outlier-std-ratio 2.0 --completion-rounds 1 --completion-max-distance 0
```

当前配置文件中的 Poisson 噪点清理默认值为很轻的清理，优先保留点云结构：

```text
statistical_outlier_nb_neighbors = 10
statistical_outlier_std_ratio = 3.0
radius_outlier_nb_points = 0
radius_outlier_radius = 0.0
```

如果噪点仍然多，可以把 `statistical_outlier_std_ratio` 降到 `1.6` 或 `1.2`，并开启半径离群点去除，例如 `radius_outlier_nb_points = 4`、`radius_outlier_radius = 0.12`。

适用场景：

适合重建较完整、较密集的点云表面。对于兔子模型，通常优先尝试 Poisson。

如果程序停在 `Poisson 重建` 阶段，常见原因是点云太大、`depth` 太高或没有下采样。二维深度图 `.npy` 会先转换成点云，例如 `480 x 640` 深度图最多会产生约 307200 个有效像素点；如果很慢，可以增大 `voxel_size`：

```powershell
.\.venv\Scripts\python.exe poisson_reconstruction.py --config reconstruction_config.json --voxel-size 0.1
```

### Ball Pivoting

配置运行：

```powershell
.\.venv\Scripts\python.exe ball_pivoting_reconstruction.py --config reconstruction_config.json
```

直接运行：

```powershell
.\.venv\Scripts\python.exe ball_pivoting_reconstruction.py --input object.xyz --output object_ball_pivoting.obj --params-out ball_pivoting_params.json --radii 0.003,0.006,0.012,0.024
```

几何判据：

```text
对于点 pi, pj, pk 和球半径 r：
如果半径为 r 的球可以同时与三个点相切，
并且球内部不包含其他采样点，则接受三角形 (pi, pj, pk)。
```

多个半径可以让算法在不同局部尺度上连接三角面。

主要参数：

- `radii`：滚球半径列表，用英文逗号分隔。
- `normals_radius`：估计法线时的邻域半径。
- `normals_max_nn`：估计法线时最多使用的邻居数量。

适用场景：

适合干净、密集、采样比较均匀的点云。点云太稀疏或密度变化太大时容易出现孔洞。

### Alpha Shape

配置运行：

```powershell
.\.venv\Scripts\python.exe alpha_shape_reconstruction.py --config reconstruction_config.json
```

直接运行：

```powershell
.\.venv\Scripts\python.exe alpha_shape_reconstruction.py --input object.xyz --output object_alpha_shape.obj --params-out alpha_shape_params.json --alpha 0.03
```

几何判据：

```text
先构建 Delaunay 四面体剖分。
如果 simplex sigma 的外接球半径 R(sigma) <= alpha，则保留该 simplex。
保留下来的 simplex 的边界形成 alpha-shape 表面。
```

主要参数：

- `alpha`：半径阈值。值越小越能保留凹陷，但网格可能破碎；值越大越接近凸包。

适用场景：

适合通过一个全局半径阈值控制凹陷和外形的重建任务。

### Convex Hull

配置运行：

```powershell
.\.venv\Scripts\python.exe convex_hull_reconstruction.py --config reconstruction_config.json
```

直接运行：

```powershell
.\.venv\Scripts\python.exe convex_hull_reconstruction.py --input object.xyz --output object_convex_hull.obj --params-out convex_hull_params.json
```

核心公式：

```text
conv(P) = { sum_i lambda_i * p_i | p_i in P, lambda_i >= 0, sum_i lambda_i = 1 }
```

凸包是包含所有输入点的最小凸集合。

主要参数：

- `joggle_inputs`：对接近退化的点做轻微扰动，帮助 QHull 处理困难输入。

适用场景：

适合快速生成外轮廓、包围体或做结果检查。它不能表示凹陷、孔洞和复杂内部结构。

### Marching Cubes

配置运行：

```powershell
.\.venv\Scripts\python.exe marching_cubes_reconstruction.py --config reconstruction_config.json
```

直接运行：

```powershell
.\.venv\Scripts\python.exe marching_cubes_reconstruction.py --input object.xyz --output object_marching_cubes.obj --params-out marching_cubes_params.json --grid-resolution 64 --ball-radius 0.006
```

核心公式：

```text
F(x) = min_i || x - p_i || - r
surface = { x | F(x) = level }
```

当前实现会把点云转换成“点球并集”的隐式标量场，再用 Marching Cubes 提取等值面。

主要参数：

- `grid_resolution`：每个坐标轴上的体素网格采样数量。
- `ball_radius`：每个点周围的小球半径。越大越容易闭合孔洞，但细节会变模糊。
- `padding`：包围盒外扩比例，按点云对角线计算。
- `level`：等值面值，通常使用 `0.0`。

适用场景：

适合体素/隐式场风格的实验性重建，也适合观察点云在不同半径下形成的包络面。

## 方法选择建议

- 想要简单、可解释的参数化近似：使用 Superquadric。
- 想让兔子尽量像兔子：优先使用 Poisson。
- 点云干净、密集、采样均匀：尝试 Ball Pivoting。
- 想通过半径阈值控制凹陷：尝试 Alpha Shape。
- 只需要最外层包围壳：使用 Convex Hull。
- 想做体素或隐式场重建实验：使用 Marching Cubes。

## 螺栓模型生成

`generate_bolt_model.py` 可以直接生成一个带六角头的外螺纹螺栓模型，以及一个配套的六角内螺纹螺母。默认参数为：

```text
螺纹数量：20
螺纹杆长度：2 cm，也就是 20 mm
螺纹外径：0.5 cm，也就是 5 mm
配套螺母高度：0.5 cm，也就是 5 mm
螺母打印间隙：0.15 mm
输出单位：毫米
```

生成命令：

```powershell
.\.venv\Scripts\python.exe generate_bolt_model.py
```

输出文件：

```text
model_outputs/bolt_20_threads/bolt_20_threads_2cm_x_0_5cm.obj
model_outputs/bolt_20_threads/bolt_20_threads_2cm_x_0_5cm.stl
model_outputs/bolt_20_threads/matching_nut_for_bolt_20_threads.obj
model_outputs/bolt_20_threads/matching_nut_for_bolt_20_threads.stl
model_outputs/bolt_20_threads/bolt_20_threads_2cm_x_0_5cm_blender_meters.obj
model_outputs/bolt_20_threads/bolt_20_threads_2cm_x_0_5cm_blender_meters.stl
model_outputs/bolt_20_threads/matching_nut_for_bolt_20_threads_blender_meters.obj
model_outputs/bolt_20_threads/matching_nut_for_bolt_20_threads_blender_meters.stl
model_outputs/bolt_20_threads/bolt_and_nut_dimensions_mm.json
```

OBJ/STL 格式本身不强制记录单位，所以脚本会输出两套文件：

```text
普通文件：坐标数值按毫米写出，适合 3D 打印切片软件。
带 _blender_meters 的文件：坐标数值按米写出，适合 Blender 默认米制场景直接查看真实尺寸。
```

Blender 中建议导入带 `_blender_meters` 的文件。如果导入普通文件，需要把导入缩放设为 `0.001`，否则 Blender 会把 `7.8` 当成 `7.8 m`，看起来会大 1000 倍。

脚本会额外输出 `bolt_and_nut_dimensions_mm.json`，用于检查实际包围盒尺寸。可调整参数：

```powershell
.\.venv\Scripts\python.exe generate_bolt_model.py --thread-count 20 --length-cm 2 --width-cm 0.5 --nut-height-cm 0.5 --clearance-mm 0.15 --radial-segments 160 --axial-segments-per-thread 16
```

如果 3D 打印后螺母拧不上，可以把 `--clearance-mm` 增大到 `0.2` 或 `0.25`；如果间隙太松，可以减小到 `0.1`。

## VRCNet 预训练模型

MVP 数据集体积较大，不适合在本机完整训练时，可以只下载 VRCNet 官方预训练权重。当前已删除 `VRCNet/data` 下已下载的 `.h5` 数据集文件，仅保留下载脚本。

预训练权重位置：

```text
VRCNet/pretrained/extracted/pretrained_vrcnet_2048.pth
```

`VRCNet/cfgs/vrcnet.yaml` 中的 `load_model` 已设置为：

```yaml
load_model: pretrained/extracted/pretrained_vrcnet_2048.pth
```

进入 VRCNet 目录后可使用该配置进行测试：

```powershell
cd VRCNet
..\.venv\Scripts\python.exe test.py -c cfgs\vrcnet.yaml
```

如果要对单个本地点云做补全，可以使用项目根目录的 `vrcnet_complete_point_cloud.py`。它会把点云采样到 2048 点，归一化后送入预训练 VRCNet，再把输出反归一化成原始尺度：

```powershell
.\.venv\Scripts\python.exe vrcnet_complete_point_cloud.py --input input\livox_raw_120deg_500k_20260511_121137.pcd --partial-out model_outputs\vrcnet_livox\livox_partial_2048.ply --output model_outputs\vrcnet_livox\livox_vrcnet_completed.ply --voxel-size 0.05
```

注意：VRCNet 原代码依赖 CUDA 版本的 PointNet++ / EMD 等算子，因此预训练推理通常需要可用 NVIDIA CUDA 环境。如果没有 CUDA，可以继续使用本项目 `poisson_reconstruction.py` 里的局部插值式补全，但那不是 VRCNet 模型补全。

当前 Windows 环境没有本地 `nvcc` / MSVC 编译环境，因此项目为单点云推理增加了轻量 fallback：

```text
VRCNet/utils/Pointnet2.PyTorch/pointnet2/pointnet2_cuda.py
VRCNet/utils/emd/emd.py
VRCNet/utils/ChamferDistancePytorch/chamfer3D/dist_chamfer_3D.py
```

`vrcnet_complete_point_cloud.py` 会跳过 EMD/Chamfer 评测指标，只输出补全点云。该路径适合推理，不适合重新训练或论文评测。

对于 Livox、教室、室内大场景点云，不建议直接使用 VRCNet 这种单物体补全网络。更稳的做法是使用场景几何补全：

```powershell
.\.venv\Scripts\python.exe scene_point_cloud_completion.py --input input\livox_raw_120deg_500k_20260511_121137.pcd --output model_outputs\scene_livox_completion\livox_scene_completed.ply --voxel-size 0.05 --statistical-nb-neighbors 10 --statistical-std-ratio 3.0 --upsample-factor 2 --k-neighbors 24 --radius 0.2 --max-source-points 60000
```

查看补全前后：

```powershell
.\.venv\Scripts\python.exe view_pcd.py input\livox_raw_120deg_500k_20260511_121137.pcd --point-size 2
.\.venv\Scripts\python.exe view_pcd.py model_outputs\scene_livox_completion\livox_scene_completed.ply --point-size 2
```

对补全后的场景点云做 Poisson 重建：

```powershell
.\.venv\Scripts\python.exe poisson_reconstruction.py --input model_outputs\scene_livox_completion\livox_scene_completed.ply --output livox_scene_completed_poisson.obj --params-out livox_scene_completed_poisson.json --voxel-size 0 --depth 10 --scale 1.1 --normals-radius 0.08 --normals-max-nn 60 --orient-max-nn 120 --density-quantile 0.02 --statistical-outlier-nb-neighbors 0 --completion-rounds 0
```

## Alpha 参数训练/寻优

如果有多份点云，可以用 `train_alpha_parameters.py` 自动寻找 Alpha Shape 的较优 `alpha`。当前实现不是神经网络监督训练，而是“候选参数搜索 + 多点云平均”：

```text
对每份点云：
  遍历 alpha_candidates
  用每个 alpha 重建 Alpha Shape 网格
  用 Chamfer 风格距离评价 点云 <-> 网格采样点 的贴合程度
  对过多连通分量增加惩罚
  选出该点云最佳 alpha

最终 recommended_alpha = mean(每份点云的 best_alpha)
```

评分近似为：

```text
score = mean distance(points -> mesh_samples) / scale
      + mean distance(mesh_samples -> points) / scale
      + max(0, component_count - 1) * component_penalty
```

训练配置位于 `reconstruction_config.json`：

```json
"training": {
  "alpha_shape": {
    "inputs": ["object.xyz"],
    "alpha_candidates": [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.07],
    "sample_points": 4000,
    "max_eval_points": 5000,
    "component_penalty": 0.03
  }
}
```

只输出训练结果，不修改默认配置：

```powershell
.\.venv\Scripts\python.exe train_alpha_parameters.py --config reconstruction_config.json
```

训练后自动把推荐值写回 `methods.alpha_shape.alpha`：

```powershell
.\.venv\Scripts\python.exe train_alpha_parameters.py --config reconstruction_config.json --update-config
```

直接指定多份点云和候选值：

```powershell
.\.venv\Scripts\python.exe train_alpha_parameters.py --inputs object1.xyz object2.xyz object3.xyz --alpha-values 0.01,0.02,0.03,0.04 --update-config
```

如果以后有真实网格作为标签，可以再扩展成真正的监督训练，例如用全连接网络根据点云统计特征预测 `alpha`。当前版本更适合样本较少的情况，结果也更可解释。

## Poisson 参数训练/寻优

`train_poisson_parameters.py` 用原始点云和重建后的 Poisson 网格之间的差距进行自监督打分，自动搜索较优参数组合。它不是监督训练，因为没有人工标注的最佳参数或真实标准网格。

训练时会按候选参数组合显示进度，例如当前正在评估第几组参数。

打分方式改为更接近点云重建论文中的常用评测指标。因为当前没有真实标准网格，脚本把原始点云视为观测参考，把重建网格表面采样点视为重建结果：

```text
completeness = mean distance(input_points -> mesh_samples) / scale
accuracy     = mean distance(mesh_samples -> input_points) / scale
chamfer_l1   = accuracy + completeness

precision = ratio(distance(mesh_samples -> input_points) / scale <= threshold)
recall    = ratio(distance(input_points -> mesh_samples) / scale <= threshold)
f_score   = 2 * precision * recall / (precision + recall)

score = chamfer_l1
      + (1 - f_score) * f_score_weight
      + max(0, component_count - 1) * component_penalty
```

含义：

- `accuracy`：重建网格表面到原始点云的平均距离，越小表示网格越少偏离观测点。
- `completeness`：原始点云到重建网格表面的平均距离，越小表示原始点云越多被网格覆盖。
- `chamfer_l1`：论文中常见的 Chamfer Distance 风格双向距离。
- `precision`：重建网格采样点中，有多少落在距离阈值内。
- `recall`：原始点云点中，有多少能被重建网格在距离阈值内覆盖。
- `f_score`：`precision` 和 `recall` 的调和平均，越接近 `1` 越好。
- `component_count`：网格碎成多个连通块时增加惩罚。
- `scale`：点云包围盒对角线，用来归一化不同大小的点云。
- `f_score_threshold`：F-score 的归一化距离阈值，默认 `0.01`，表示点云包围盒对角线的 1%。
- `f_score_weight`：`1 - f_score` 在最终最小化分数中的权重。

训练配置位于 `reconstruction_config.json`：

```json
"training": {
  "poisson": {
    "inputs": ["input/livox_raw_120deg_500k_20260511_121137.pcd"],
    "depth_values": [7, 8, 9, 10],
    "scale_values": [1.05, 1.1],
    "normals_radius_values": [0.02, 0.05],
    "normals_max_nn_values": [30, 50],
    "orient_max_nn_values": [50, 100],
    "density_quantile_values": [0.0, 0.02, 0.05],
    "linear_fit_values": [false],
    "sample_points": 4000,
    "max_eval_points": 5000,
    "voxel_size": 0.05,
    "max_reconstruction_points": 20000,
    "component_penalty": 0.03,
    "f_score_threshold": 0.01,
    "f_score_weight": 0.05
  }
}
```

只输出训练结果，不修改默认配置：

```powershell
.\.venv\Scripts\python.exe train_poisson_parameters.py --config reconstruction_config.json
```

训练后自动写回 `methods.poisson` 的默认参数：

```powershell
.\.venv\Scripts\python.exe train_poisson_parameters.py --config reconstruction_config.json --update-config
```

为了快速试验，可以在命令行缩小候选范围：

```powershell
.\.venv\Scripts\python.exe train_poisson_parameters.py --inputs input\livox_raw_120deg_500k_20260511_121137.pcd --depth-values 8,9 --normals-radius-values 0.03,0.05 --normals-max-nn-values 30,50 --orient-max-nn-values 50,100 --density-quantile-values 0.02,0.05 --voxel-size 0.05 --max-reconstruction-points 20000 --f-score-threshold 0.01 --f-score-weight 0.05 --update-config
```

对于几十万点的 PCD，训练时建议使用 `--voxel-size` 或 `--max-reconstruction-points` 先下采样，否则每组 Poisson 参数都要对全量点云重建，耗时会非常长。训练得到参数后，可以再用 `poisson_reconstruction.py` 对原始点云做正式重建。

### 粗到细搜索 vs Optuna 对比

如果想同时试两种训练方向，可以运行：

```powershell
.\.venv\Scripts\python.exe train_poisson_compare.py --fine-rounds 1 --optuna-trials 8 --base-args --inputs input\livox_raw_120deg_500k_20260511_121137.pcd --depth-values 7,8 --scale-values 1.05,1.1 --normals-radius-values 0.03,0.05 --normals-max-nn-values 30,50 --orient-max-nn-values 50,100 --density-quantile-values 0.02,0.05 --voxel-size 0.1 --max-reconstruction-points 8000 --sample-points 1500 --max-eval-points 2000 --f-score-threshold 0.01 --f-score-weight 0.05
```

这个脚本会输出两套配置：

```text
model_outputs/training/poisson_compare/poisson_coarse_to_fine_config.json
model_outputs/training/poisson_compare/poisson_optuna_config.json
```

再分别生成两个网格：

```powershell
.\.venv\Scripts\python.exe poisson_reconstruction.py --config model_outputs\training\poisson_compare\poisson_coarse_to_fine_config.json
.\.venv\Scripts\python.exe poisson_reconstruction.py --config model_outputs\training\poisson_compare\poisson_optuna_config.json
```

输出会分别使用：

```text
poisson_coarse_to_fine.obj / poisson_coarse_to_fine.ply
poisson_optuna.obj / poisson_optuna.ply
```

你可以打开这两个网格，人工判断哪种参数训练方向更好。

## 局限性

没有一种方法适合所有点云。点云稀疏、噪声大、法线不稳定、密度不均匀、扫描视角不完整，都会导致孔洞、漂浮面片、过度平滑或外形错误。

建议先修改 `reconstruction_config.json` 中对应方法的参数，再重新运行对应脚本。
