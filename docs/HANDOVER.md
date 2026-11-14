# zmocap 接手审计与交接文档

> 审计日期：2026-11-11
> 代码版本：`ba128da`（`main` == `origin/main`，工作区干净）
> 审计范围：仓库内 49 个受版本控制的文件 + 本机运行环境 + 现有产物
> **审计阶段未修改代码；2026-11-11 21:10 按用户要求修复了 SMPL Previewer
> （GLB 索引/UV accessor 计数）并更新了桌面工作流，详见文末“后续修复”。** 下文所有数字都可用第 9 节的命令复现。

---

## 0. 一页结论

### 0.1 这个项目是什么

在 ComfyUI 里把**视频**变成**世界坐标 3D 骨骼动画**，并绑定到 VRM/GLB 角色上预览、导出。
算法内核是 vendored 的 ExMR（Sapiens2 关键点 + SAM-3D-Body 特征 + HaMeR 手部特征 → SMPL-X），
外面用 ComfyUI 节点重新组织；转换/绑定数学是本仓库自己写的。

### 0.2 健康度判断

**能跑通，但“能跑”不等于“正确”。**

- ExMR 推理链在本机（8 GB 显存）确实端到端跑通了：日志里 107 帧约 90–100 秒，243 帧约 2 分钟。
- 两条绑定链里，**只有 SMPL → 角色那条数学是闭环的**（静止姿态保真 0.0°，单测 4e-6°）。
- **SOMA → VRM → 角色那条（也就是 DESIGN 主推的路线）有一个系统性错误：**
  把“静止动作”喂进去，输出角色会被拧 154°，52 根骨里 42 根偏移 >5°。
  根因是 `soma_to_vrm()` 没有把源运动归一化到标准骨架的静止基（§4 P0-2，已用原型验证修法能把误差降到 0.000°）。
- 另外还有三个独立问题会直接伤到用户：导出 BVH 是非法文件、SMPL-X 前向的手部约定错
  （最大 45.5 mm）、小指（SOMA 的 Pinky 链）完全不参与动画。
- **仓库当前不可复现**：`vendor/` 被 `.gitignore` 排除，全新 clone 无法运行 ExMR 和 VRM 绑定；
  `requirements.txt` 全是注释。发布前必须解决。

### 0.3 五个必须知道的事实

1. **正在跑的 ComfyUI 是旧代码。** 服务 PID 27649 于 17:56:48 启动，但 7 个源文件在那之后被修改/提交
   （18:01–19:15）。在 8188 上看到的行为**不是**磁盘上 `ba128da` 的行为，验证任何修复前先重启。
2. **`vendor/` 不在 git 里**（`git ls-tree origin/main vendor/` 为 0 个文件）。ExMR 代码、MotionBridge 代码都是本机文件；
   新用户 clone 下来只能 `import nodes`，跑不了 ExMR，`soma.binding` 也直接 `ModuleNotFoundError`。
3. **SOMA→VRM 路径的静止姿态不保真**（§4 P0-2）。这是当前最值得先修的正确性问题。
4. **两条链的能力不对称**：`nodes/retarget.py` 的 SMPL→角色链支持**无冒号**的 Mixamo 命名
   （`mixamorigLeftArm`），但 `mixamorig:LeftArm` 也不命中；`soma/binding.py` 的 SOMA→VRM 链
   只认 VRM `J_Bip_*` 和 `l_/r_` 命名，**Mixamo 名称 0/20 命中**。
5. **导出节点目前只对 SMPL 动作用得了**：`ZMocapExport` 吃 `motion["joints"]`，接 SOMA/VRM 动作直接 `KeyError: 'joints'`；
   即使接 SMPL 动作，写出的 BVH 也是非法的（声明 168 个通道/帧，实际只写 6 个值/帧，且只写 55 个关节）。

### 0.4 建议的接手顺序

```
重启服务 → 固化基线（单测 + 静止不变式）→ 修 P0-2（rest 归一化）
→ 修四元数分支 → 修 hands_mean → 修 Pinky/Mixamo 映射
→ 修导出 → 修 install/workflow/vendor/requirements → 再谈发布
```

---

## 1. 审计范围与方法

做了什么：

- 通读全部 `nodes/`、`soma/`、`support/`、`tools/`、`scripts/`、`web/js/` 代码（vendor 只读了入口与许可）。
- `ruff check --select F,E9` 静态检查（发现 12 项，见 §4 P3）。
- 运行现有单测 `python -m zmocap.tools.test_retarget`。
- 用**现有真实产物**做数值回归：
  - `ComfyUI/output/zmocap_motion.npz`：107 帧、127 关节的 ExMR 输出；
  - `ComfyUI/output/zmocap_character_Sendagaya_Shino.glb`：52 根 humanoid 骨的 VRM 角色；
  - 用 `tools/fk_check.py` 的 GLB 采样器做前向运动学对照。
- 用 `/object_info` 检查正在运行的服务注册了哪些节点（16 个 zmocap 节点）。
- 用 `git archive HEAD` 模拟“全新 clone”验证可复现性。
- 没有做的事：没有跑 GPU 全链路（避免干扰正在使用的 8188 服务）；没有浏览器实测前端（只做了语法检查）；
  没有在 Windows/macOS/非 NVIDIA 上验证。

---

## 2. 项目速览

### 2.1 两条处理链

```
视频
 └─(官方 LoadVideo / GetVideoComponents)─ IMAGE
      ├─ ZMocapTrack        → tracks(框 + 内参 + 画布)
      ├─ ZMocapFrames       → 帧目录(/tmp)
      └─ ZMocapModels       → models 根目录

  tracks + frames + models
      ├─ ZMocapSapiens2Keypoints  → (T,63,3)
      ├─ ZMocapSam3DBodyFeatures  → (T,1024)
      └─ ZMocapHaMeRHandFeatures  → (T,2,1024)
                    ↓
              ZMocapExMR  → motion(smplx)：params + joints(127) + fps
                    │
        ┌───────────┴─────────────────────────────┐
        │                                         │
  ZMocapSMPLPreviewer                       ZMocapSMPL2SOMA
  （SMPL-X 自带网格，零重定向，            → motion(soma, 78 关节)
    可选绑定角色 build_glb）                      ↓
        │                                  ZMocapSOMA2VRM
        │                                  → motion(vrm, 52 骨)
        │                                         ↓
        │                                  ZMocapVRMPreviewer ← ZMocapVRMLoader
        │                                  （绑角色 / 无角色标准骨架）
        └────────────── ZMocapExport（NPZ + BVH，仅 SMPL 动作可用）
```

约定：`motion["skeleton"] ∈ {smplx, soma, vrm}`；转换节点只换表示，绑定/预览节点才碰角色。

### 2.2 数据契约（实际实现）

| 类型 | 形状/结构 | 产出节点 |
|---|---|---|
| `ZMOCAP_TRACKS` | `boxes (T,4)`、`K (T,3,3)`、`width/height`、`observed`、`track_ids` | Track / Intrinsics |
| `ZMOCAP_FRAMES` | `dir`、`count`、`width/height`、`token` | Frames |
| `ZMOCAP_KEYPOINTS` | `xy (T,63,3)`、`count` | Sapiens2 |
| `ZMOCAP_BODY_FEATURES` | `values (T,1024)` | SAM3D |
| `ZMOCAP_HAND_FEATURES` | `values (T,2,1024)` | HaMeR |
| `ZMOCAP_MOTION(smplx)` | `params{betas,global_orient,body_pose,left/right_hand_pose,transl}`（torch CPU）、`joints (T,127,3)`、`fps`、`source` | ExMR |
| `ZMOCAP_MOTION(soma)` | `names 78`、`parents`、`rest (78,4,4)`、`world (T,78,4,4)`、`fps` | SMPL2SOMA |
| `ZMOCAP_MOTION(vrm)` | `names 52`、`parents`、`rest (52,4,4)`、`world_rotation (T,52,3,3)`、`transl (T,3)`、`fps` | SOMA2VRM |
| `ZMOCAP_CHARACTER` | `{path,name,preview,nodes,meshes,bones}` | VRM Loader |
| `ZMOCAP_MODELS` | `{root,vendor,low_memory}`（`low_memory` 实际无人使用） | Models |

### 2.3 代码地图

| 路径 | 作用 | 备注 |
|---|---|---|
| `__init__.py` | ComfyUI 入口，`WEB_DIRECTORY="./web/js"` | |
| `nodes/__init__.py` | 16 个节点注册 + 显示名 | 没有注册 `ZMocapRig` |
| `nodes/exmr_engine.py` | vendored ExMR 胶水：模型链接、加载、三阶段、推理 | 直接 `.cuda()`，不走 `comfy.model_management` |
| `nodes/exmr.py` | ExMR 节点 + SMPL-X 前向出 127 关节 | `_joints()` 每次新建 SMPLX |
| `nodes/track.py` | YOLO 检测 + 贪心 IoU 关联 | 每帧只保留**最大**框 |
| `nodes/frames*.py` | IMAGE ⇄ 帧目录、内参 | 临时目录从不清理 |
| `nodes/retarget.py` | SMPL→角色 GLB；GLB 读写/四元数/动画写出 | 829 行，含未注册的 `ZMocapRig` 与重复定义 |
| `nodes/smpl_mesh.py` | SMPL-X 参数 → 蒙皮 GLB（对照基准） | 数值上可信 |
| `nodes/sticks.py` | 骨骼小棍图层（挂在骨骼节点下） | 当前可用（1e52895 之后） |
| `nodes/character.py` | VRM/GLB 载入、下拉框、复制到 output | 实际只支持 GLB 容器 |
| `nodes/previewer.py` | SMPL / VRM 两个预览器 | 输出 `ui.zmocap_glb` 给前端 |
| `soma/motion.py` | SMPL-X params → SOMA 78 关节 | |
| `soma/vrm_motion.py` | SOMA → 标准 VRM humanoid 运动（**P0-2 所在**） | |
| `soma/binding.py` | humanoid 运动 + 角色 → 世界旋转；写 GLB | 名称映射只覆盖 VRM |
| `soma/vrm.py` | 角色骨架转储、GLB 动画写出、旧 MotionBridge 重定向 | 旧路径已死，但仍 import vendor |
| `soma/humanoid.py` | 标准 52 骨骼（名字/父子/rest） | 有死循环、叶子骨方向默认 +Y |
| `web/js/zmocap.js` | 节点内 3D 预览、上传按钮、图层开关 | 默认角色值判断有 bug |
| `web/js/zmocap_viewer.js` | three.js 渲染/动画/图层/相机 | 只切换已有对象，不画骨骼 |
| `install.py` | 模型自检 | 路径/阈值与实际不符 |
| `scripts/make_workflow.py` | 生成工作流模板 | 与当前节点图脱节 |
| `tools/test_retarget.py` | 唯一的单测（SMPL 链） | 覆盖不到 quat 分支 |

### 2.4 本机部署事实

| 项 | 值 |
|---|---|
| ComfyUI | `/home/tony/zrepo/ComfyUI`，0.41.0 |
| Python | `/home/tony/zvenv/comfyui/bin/python`，3.13.5 |
| torch | 2.13.0+cu130，CUDA 可用 |
| GPU | 8 GB（日志 `Total VRAM 8188 MB`），RAM 32 GB |
| custom node | `/home/tony/zrepo/ComfyUI/custom_nodes/zmocap` → `/home/tony/zrepo/zmocap`（软链） |
| 模型 | `ComfyUI/models/zmocap` → `/home/tony/zext/comfyui/models/zmocap`；内部 `body_models/exmr/flowhmr/handflow/yolo` 再软链到 `/mnt/b/RGBMoCap/models` |
| 角色 | `models/zmocap/characters/{fem_vroid,Sendagaya_Shino}.vrm`；`ComfyUI/input/` 里也各有一份 |
| 服务 | `127.0.0.1:8188`，PID 27649，启动于 2026-11-11 17:56:48 |
| 日志 | `ComfyUI/user/comfyui_8188.log`（另有 `.prev*`） |

**服务新旧判断**：`git log` 显示 18:01、18:34、18:58、19:03、19:15 都有提交；`find -newermt '17:56'` 命中
`scripts/make_workflow.py`、`soma/binding.py`、`soma/vrm.py`、`nodes/sticks.py`、`nodes/previewer.py`、
`nodes/smpl_mesh.py`、`nodes/__init__.py`。所以 8188 上的 16 个节点是 17:56 的代码，**不含最新预览器拆分与 SMPL 网格预览**。

---

## 3. 现在能跑通什么（实测证据）

### 3.1 ExMR 端到端

日志（`ComfyUI/user/comfyui_8188.log`）显示 107 帧完整链：

```
frames 107 @ 640x640
sapiens2 keypoints (107, 63, 3)
hamer hand features (107, 2, 1024)
sam3d body features (107, 1024)
SMPL2SOMA: 107 帧 × 78 关节
SOMA2VRM: 107 帧 × 52 骨
VRM preview zmocap_vrm_xxx.glb (52 bones, scale 1.0)
Prompt executed in 101.21 seconds
```

243 帧版本在 18:41–18:43 也跑通，约 2 分钟。说明单环境 + 8 GB 显存可跑，模型软链方案有效。

### 3.2 SMPL 网格预览是可信的“标准答案”

用 107 帧产物重建 GLB，并做 FK 对照（`joints` vs GLB 节点世界位置）：

| 部位 | 平均误差 | 最大误差 |
|---|---:|---:|
| 躯干/四肢（0–21） | 0.000 mm | 0.000 mm |
| 下颌/眼睛（22–24） | 0.000 mm | 0.000 mm |
| 左右手（25–54） | 16.9 mm | **45.5 mm** |

手部误差的根因不是网格，而是 `nodes/exmr.py` 里 `_joints()` 用了
`flat_hand_mean=False`，而 vendored ExMR 的 SMPL-X 模型是 `flat_hand_mean=True`
（`vendor/ExMR/exmr/body_models/smplx_model.py:33`）。把 `_joints()` 改成 `True` 后，
同一组数据最大误差 **0.000 mm**（已实测）。详见 P1-1。

### 3.3 SMPL→角色链的数学是对的

- `python -m zmocap.tools.test_retarget`：`最大世界朝向误差 4.005e-06°`，PASS。
- 用零姿态 SMPL 参数 + `build_glb(character=...)` 重放，输出节点世界旋转与角色 rest
  的最大偏差 **0.0°**（52 骨）。

结论：`nodes/retarget.py::_add_animation` 的相似变换（`D_parent⁻¹ · R_rest · D_parent · R_i`）可信。
它的问题只在四元数转换的边界分支（P1-4）和骨骼命名映射。

### 3.4 SOMA→VRM 链的“静止不变式”失败

用零姿态 SMPL-X 参数走完整链（SMPL2SOMA → SOMA2VRM → bind → 写 GLB），
与角色原始 rest 对比：

```
最大偏差 154.29°   中位 11.04°   >5° 的骨骼 42/52
最差：左右脚 149–154°，左右脚趾 146–151°，左右拇指末节 108°，
      hips 104.6°，左右手 90–94°，左右 index/middle/ring 末节 94–96°
```

这不是“骨骼轴约定不同”能解释的：**输入是零姿态，输出应当原样回到角色 rest**。
根因与修法见 P0-2。

### 3.5 其它已验证的现状

- 骨骼小棍：用当前磁盘代码重新绑一次，输出 `加骨骼小棍 52 根`，`animations=['zmocap']`，
  小棍作为骨骼节点子节点跟随动画（1e52895 的修法有效；日志里的
  `小棍图层失败（'list' object has no attribute 'get'）` 是旧进程的旧代码）。
- VRM 角色载入：`Sendagaya_Shino.vrm` 读出 153 nodes / 3 meshes / 52 骨，复制到 output 成功。
- 前端 JS：`node --check` 通过；`GLTFLoader.js`/`OrbitControls.js` 的相对 import 正确。
- `web/js` 预览器会播放 GLB 里的**所有**动画（`for (const clip of gltf.animations) play()`），
  所以“蒙皮动画 + 小棍动画”两条 animation 都能动。

---

## 4. 缺陷清单（按优先级）

> 编号用于后续 issue/提交引用。P0 = 阻断发布/接手；P1 = 功能错误；P2 = 可用性/一致性；P3 = 卫生。

### P0-1 `vendor/` 不在版本库，项目不可复现

**现象**

- `git ls-tree -r origin/main --name-only | grep '^vendor/'` → **0**。
- `.gitignore` 第 4 行 `/vendor/`；`vendor/ExMR`（12 MB）、`vendor/motionbridge`（192 KB）、
  `vendor/KimodoMotionCorrection`（908 KB，**完全未使用**）都只在本机。
- 全新 clone 模拟：
  - `import zmocap.nodes` 成功（16 个节点）；
  - `import zmocap.soma.binding` → `ModuleNotFoundError: No module named 'zmocap_pkg.vendor'`；
  - `build_assets()` → `RuntimeError: ExMR 代码缺失: .../vendor/ExMR/exmr（vendor/ExMR 应随仓库部署）`。

**影响**

新用户/CI/另一台机器上：ExMR 链跑不了，VRM 预览器只要碰到 `soma.binding` 就 import 失败。
README 的安装步骤完全没提 vendor，等于“clone 下来不可用”。

**加重因素**

- `soma/vrm.py:14` 顶层 `from ..vendor.motionbridge import rigmap, retarget`；但实际在用的
  `binding.py` 只调用 `build_dump` / `write_animation`，`retarget_to_rig` 是死代码。
  也就是说：**一个死代码 import 把整个 VRM 路径硬绑到了 vendor**。
- `vendor/__init__.py` 本身也不在 git 里（本机 16:03 才创建）；即使手动拷 vendor 进去，
  没有 `__init__.py` 相对 import 仍会失败。

**建议**

1. 选一种可复现方案：
   - (a) 直接把 `vendor/ExMR` + `vendor/motionbridge` 提交进仓库（12 MB 可接受），
     同时提交 `vendor/__init__.py`、各自 LICENSE/NOTICE；
   - (b) 用 git submodule 固定上游 commit；
   - (c) 写 `scripts/fetch_vendor.py`，按固定 commit 下载并校验 sha256，README 写清。
2. 删除 `soma/vrm.py` 里对 vendor 的顶层依赖（把 `retarget_to_rig` 及 vendor import 删掉，
   只保留 `_read_glb`/`_Appender` 等本地工具），让“不用旧算法”的路径不再依赖 MotionBridge。
3. `NOTICE` 要补 MotionBridge 的许可证；目前 `vendor/motionbridge` 下没有任何 LICENSE/NOTICE，
   而 `soma/vrm.py` 声称 Apache-2.0，属于许可不合规风险。

---

### P0-2 SOMA→VRM 没有做“静止归一化”，静止输入会把角色拧 154°

**现象**

零姿态输入（所有 pose 参数为 0）走完整链，输出与角色 rest 的最大偏差 154.29°，42/52 骨 >5°（§3.4）。

**根因**

`soma/vrm_motion.py::soma_to_vrm()` 用**源的绝对朝向**构造标准骨架的 `world_rotation`：

```python
world_rotation[:, index] = [xaxis, directions, zaxis]
```

其中 `directions` 来自 SOMA 关节位置（SMPL-X rest 的骨向）。而 `soma/binding.py` 的绑定公式是

```
W_char(t) = A(t) @ B_can⁻¹ @ B_char
```

这个公式成立的前提是 **A(rest) == B_can**。但 `soma_to_vrm` 给出的 A(rest) 是
**SMPL-X 的 rest 基**，不等于 `humanoid.rest_matrices()`（标准 VRM rest）。两者在脚、脚趾、
髋、手、手指末节上差 90–154°。于是零姿态输入被当成“源 rest 相对标准 rest 的偏移”，
角色被强行拧成 SMPL-X 的 rest。

**为什么 SMPL 链没这个问题**

`_add_animation` 的输入是 SMPL-X 的**局部增量** `R_i`，公式在 `R_i=I, D=I` 时严格退化为
`L_i = R_rest`，所以静止保真。SOMA 链输入的是绝对朝向，缺了“除以源 rest”这一步。

**已验证的修法（原型，未落盘）**

把 `soma_to_vrm` 里构造朝向的循环抽成 `_orientations(positions, world_rotations)`，
对 `motion.rest` 的位置 + 单位旋转再跑一次得到 `A_rest`，然后：

```python
delta = A @ A_rest.transpose(0, 2, 1)          # 源的世界增量
world_rotation = delta @ rest[:, :3, :3]       # 叠到标准 rest 上
```

等价写法：`A_norm = A @ A_rest.T @ B_can`，之后 `W_char = A_norm @ B_can.T @ B_char`
（`B_can` 会抵消，最终是 `A @ A_rest.T @ B_char`）。

实测结果：

```
零姿态静态偏差：max 0.000°  median 0.000°  >5° 0/52
真实动作增量（W(t)W(0)ᵀ vs A(t)A(0)ᵀ）：max 0.0013°  median 0.0007°（与 quat 修复一起验证）
```

**验收标准**

- 零姿态输入 → 输出与角色 rest 逐骨误差 < 0.01°。
- 真实动作：世界旋转增量 `W(t)W(0)ᵀ` 与目标增量 `A(t)A(0)ᵀ` 逐骨误差 < 0.01°（quat 修复后）。

**注意**

- 修完后 `humanoid.rest_matrices()` 的解剖学小错误（hips 选最长子骨、叶子骨默认 +Y）
  对绑定不再有影响（会被抵消），但仍影响标准骨架预览，建议一并修（P2-5）。
- `write_humanoid_glb` 用同一套 `rest_matrices`；修 rest 归一化不会让它变坏。

---

### P0-3 导出不可用：BVH 非法 + 非 SMPL 动作 KeyError

**现象 1：BVH 非法**

`output/zmocap_motion.bvh`（107 帧）解析结果：

```
ROOT=1  JOINT=54  CHANNELS 行=55
声明的通道总数 = 6 + 54×3 = 168
MOTION 每帧实际数值 = 6
Frames: 107
```

任何 BVH 解析器（Blender、BVH viewer）都会因为“通道数 ≠ 每帧数值数”直接失败；
即使能读，也只有根节点位移，所有子骨旋转恒 0，等于没有动画。
另外 `SMPLX_PARENTS` 只有 55 项，`joints` 是 127 个，手/脸被静默丢弃。

**现象 2：接 SOMA/VRM 动作 KeyError**

`ZMocapExport.run()` 第 36 行：

```python
joints = np.asarray(motion["joints"], dtype=np.float32)
```

SOMA/VRM 动作没有 `joints` 键（只有 `world` 或 `world_rotation`），实测：

```
EXPORT ERROR: KeyError 'joints'
```

而工作流文档又建议 SMPL2SOMA → SOMA2VRM → … 之后导出，用户一定会踩。

**建议**

1. 拆成两个导出器或在节点内按 `motion["skeleton"]` 分支：
   - `smplx`：`joints + params`；
   - `vrm`：`world_rotation + transl + rest`；
   - `soma`：`world + rest`。
2. BVH 用 `local_rotations(params)` 生成每关节局部旋转，转成 BVH 的 `Zrotation Xrotation Yrotation`
   （或直接给每个关节 3 个 position 通道 + 根 6 通道），按声明的通道数逐帧写满。
   `SMPLX_PARENTS` 只覆盖 55 个关节，要么只导出 55 个并在 UI 说明，要么补全 127 的父表。
3. 加一个最小解析测试：写出的 BVH 用 `tools/fk_check` 或简单 parser 验证
   “通道数 == 每帧数值数”和“帧数 == Frames”。

---

### P1-1 `_joints()` 的手部约定与 ExMR 不一致（最大 45.5 mm）

**位置**：`nodes/exmr.py:96`

```python
body = smplx.SMPLX(str(body_path), flat_hand_mean=False, use_pca=False).eval()
```

**证据**

- vendored ExMR 自己的 SMPL-X：`vendor/ExMR/exmr/body_models/smplx_model.py:33` → `flat_hand_mean=True`。
- 用 GLB（由原始 params 构造）与 `smplx` 前向对比：
  - `flat_hand_mean=True`：最大 0.000 mm；
  - `flat_hand_mean=False`：手部最大 45.5 mm，body 0.000 mm。
- ExMR 输出的 `left_hand_pose/right_hand_pose` 是相对“平手”的轴角；再叠加 `hands_mean`
  等于多转了一次均值姿势。

**影响**

- `motion["joints"]` 的手部位置错误；
- SMPL 预览器的骨关节 gizmo/mannequin、`fk_check` 的对照、未来足部锁定的手部约束都会受污染。
- 注意：`local_rotations()` 和 `smpl_mesh._local_rotations()` 用的是**原始手部参数**（不加 mean），
  对当前 ExMR 约定是**正确的**；不要“顺手”给它们加 `hands_mean`。

**修法**：`flat_hand_mean=True`，并加一条“手部最大误差 < 0.01 mm”的回归测试。

---

### P1-2 小指（SOMA 的 Pinky 链）完全不参与动画

**位置**

- `soma/vrm_motion.py:36-40`：humanoid 侧叫 `Little`，SOMA 侧叫 `Pinky`：

```python
for finger in ("Index", "Middle", "Ring", "Little"):
    ...
    _BONES[f"{side}{finger}{level}"] = f"{cap}Hand{finger}{suffix}"
```

生成 `LeftHandLittle1/2/3`，但 `SOMA_NAMES` 里是 `LeftHandPinky1/2/3/4/End`。
`joint not in available` → 走 `rest` 分支 → 小指永远静止。

- `_LEAF_FALLBACK` 同样写成 `LeftHandLittleEnd`（不存在），应为 `LeftHandPinkyEnd`。

**证据**

用 107 帧真实动作跑 `soma_to_vrm`，与 rest 对比，**恰好 6 根骨全帧无变化**：

```
leftLittleProximal / Intermediate / Distal
rightLittleProximal / Intermediate / Distal
```

**修法**

```python
for finger, soma_finger in (("Index","Index"), ("Middle","Middle"),
                            ("Ring","Ring"), ("Little","Pinky")):
    for level, suffix in (("Proximal","1"), ("Intermediate","2"), ("Distal","3")):
        _BONES[f"{side}{finger}{level}"] = f"{cap}Hand{soma_finger}{suffix}"
```

并把 `_LEAF_FALLBACK` 的 `LeftHandLittleEnd` → `LeftHandPinkyEnd`（左右同理）。
验收：6 根骨在真实动作中方差 > 0；小指末节方向与源一致。

---

### P1-3 名称映射：SOMA 链对 Mixamo 0 命中，SMPL 链对冒号式 Mixamo 也不命中

**位置**：`soma/binding.py:22-50`（`_ALIAS`）与 `humanoid_of()`；`nodes/retarget.py:117`（`_normalize`）。

**证据 1：SOMA 链 0/20**

用 20 个标准 Mixamo 名称测试 `humanoid_of()`：**0/20 命中**（包括 `mixamorig:Hips`）。
原因：`_normalize` 会剥掉 `mixamorig:` 前缀，但 `_ALIAS` 的键是 `l_/r_` 前缀的 Blender 风格
和 `J_Bip_*` 归一化后的 `c/l/r` 形式，没有 “left/right” 全称键，也没有裸 `hips/spine/chest/...`：

- `mixamorig:Hips` → `hips`，但 `_ALIAS` 只有 `"chips"`，没有 `"hips"`；
- `mixamorig:LeftArm` → `leftarm`，但 `_ALIAS` 只有 `"larm"`，没有 `"leftarm"`；
- Mixamo 的 `LeftHandPinky1` → `lefthandpinky1`，也没有对应键（应映射到 `leftLittleProximal`）。

**证据 2：SMPL 链只认无冒号形式**

`nodes/retarget.py::_normalize` 只删 `_ - .`，不删 `:`，也不剥前缀。别名表里是
`mixamorigleftarm`，所以：

```
mixamorig:LeftArm   → _normalize = "mixamorig:leftarm"  → _bone_map = {}      (MISS)
mixamorigLeftArm    → _normalize = "mixamorigleftarm"   → _bone_map = {16: 0} (OK)
mixamorig:LeftUpLeg → _normalize = "mixamorig:leftupleg"→ _bone_map = {}      (MISS)
```

Mixamo FBX/GLB 的骨骼名通常带冒号（`mixamorig:LeftArm`），Blender 导出时也可能保留。
也就是说：**SMPL 路径的“支持 Mixamo”只在导出器恰好去掉冒号时成立**；
SOMA 路径则完全不支持全称命名。

**影响**

`ZMocapVRMPreviewer`/`bind` 的 UI 文案写着“支持 Mixamo / VRM 命名”，但 SOMA 路径实际只支持
VRoid/VRM `J_Bip_*`（日志里的 52/52 就是这种）。换成 Mixamo 角色时：
`world_rotation` 只映射到 hips/spine/neck/head 等少数骨（甚至 0 根），
`bind` 要么抛“没有识别到任何 humanoid 骨骼”，要么静默只绑 4 根骨。

**修法**

1. 两个 `_normalize` 都先剥掉 `mixamorig[:]`、`armature|`、`J_Bip_` 等前缀（`binding._PREFIXES` 已有，
   `retarget` 需要补），并统一小写/去分隔符。
2. 在 `_ALIAS` 里补全全称键与裸名（注意 Mixamo 的脊柱链是 Hips > Spine > Spine1 > Spine2）：

```python
_ALIAS.update({
    "hips": "hips", "pelvis": "hips",
    "spine": "spine", "spine1": "chest", "spine2": "upperChest",
    "chest": "chest", "upperchest": "upperChest", "neck": "neck", "head": "head",
})
# 左右：leftshoulder, leftarm→leftUpperArm, leftforearm→leftLowerArm,
#      leftupleg/leftupperleg→leftUpperLeg, leftleg/leftlowerleg→leftLowerLeg,
#      leftfoot, lefttoebase/lefttoes,
#      lefthandindex1..3 / lefthandmiddle / lefthandring / lefthandpinky → leftLittle*
#      lefthandthumb1..3
```

验收：上面 20 个 Mixamo 名称 20/20 命中；带冒号与不带冒号的 `mixamorig[:]LeftArm` 都能命中；
一个真实 Mixamo GLB 能绑出 52 根骨；`bind` 的 report 里 `unmapped_count == 0`。

---

### P1-4 `_quat_from_mat` 的四个分支不覆盖全部输入（约 10% 随机旋转）

**位置**：`nodes/retarget.py:301-322`；同样的实现还复制在 `tools/fk_check.py:86`。

**现象**

分支条件是：

```python
(~positive & (m00>m11) & (m00>m22), ...)
(~positive & ~(m00>m11) & (m11>m22), ...)
(~positive & ~(m00>m11) & ~(m11>m22), ...)
```

当 `trace <= 0`、`m00 > m11`、但 `m22 >= m00` 时，**三个分支都不命中**，q 保持全 0。
`_mat_from_quat` 对零四元数会归一化失败/退化成单位旋转，于是该帧该骨“啪”地回到 rest。

**证据**

20,000 个随机旋转矩阵 round-trip：

```
原实现：max error 2.000，bad(>1e-6) 2079/20000（10.4%）
修复后：max error 0.000，bad 0/20000
```

真实 107 帧动作里命中 1/5564 个局部旋转（拇指某个帧），所以表现为“偶尔抽一下”，
不是每帧都错；但对手指/脚这类小关节，一次就够明显。现有单测只覆盖了几个温和旋转，
所以 PASS 了却没发现。

**修法**

分支选择改成互斥且穷尽（trace>0 用 w 分支；否则按 m00/m11/m22 最大者）：

```python
c0 = trace > 0
c1 = (~c0) & (m00 > m11) & (m00 > m22)
c2 = (~c0) & (~c1) & (m11 > m22)
c3 = (~c0) & (~c1) & (~c2)
```

然后三个对角分支分别用 `(0,1,2)/(1,2,0)/(2,0,1)`；`tools/fk_check.py` 同步修，
或抽成 `support/quat.py` 单一实现，避免第三份拷贝。

**验收**：随机旋转 round-trip 20k 个 0 失败；真实动作的 `W(t)W(0)ᵀ == A(t)A(0)ᵀ` 逐骨 < 0.01°。

---

### P1-5 绑定的身高缩放恒为 1.0

**位置**：`soma/binding.py:71-78`（`_leg_length`）与 `105-108`。

```python
matched_heads = {name: head for name, head in bone_head.items() if humanoid_of(name)}
canonical_heads = {humanoid.NAMES[i]: humanoid.POSITIONS[i] for i in ...}
scale = _leg_length(matched_heads) / _leg_length(canonical_heads)
```

`matched_heads` 的键是**角色骨骼名**（`J_Bip_L_UpperLeg`），而 `_leg_length` 查
`heads["LeftLeg"]` / `heads["LeftFoot"]`：两边都 KeyError → 返回默认 1.0；
`canonical_heads` 的键是 `leftUpperLeg` / `leftFoot`（小写 l），也查不到。
所以 `scale` 恒为 1.0，日志里的 `scale 1.0` 不是“角色比例刚好一致”。

**影响**

根位移不按角色身高缩放；高矮差异大的角色滑步/上下浮动幅度不对。旋转不受影响。

**修法**

```python
heads_by_role = {humanoid_of(n): h for n, h in bone_head.items() if humanoid_of(n)}
def leg_length(h):
    return float(np.mean([np.linalg.norm(h[f"{s}UpperLeg"] - h[f"{s}Foot"])
                          for s in ("left", "right")]))
scale = leg_length(heads_by_role) / leg_length(canonical_heads)
```

验收：两个身高差 20% 的角色，`report["scale"]` 明显不同；根位移按比例变化。

---

### P1-6 “多人物选第 N 个”名不副实

**位置**：`nodes/track.py:69-82`。

每帧只保留**面积最大**的一个框（`ZMocapTrack._largest`），再做 IoU 关联；
`person` 只是从这个“最大框序列”里选第 N 个 track id。

**证据**（合成两个人，A 通常更大，8–11 帧被 B 超过）：

```
每帧最大框的 track id: [0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0]
每个 id 帧数: {0: 16, 1: 4}
person=1 只拿到第 8–11 帧的片段
```

`_fill_gaps` 再用最近帧的框填满整段，得到一个**不连贯**的 track。
README 写“YOLO 检测 + IoU 关联（多人物里选第 N 个）”，实际是“选最大人物的碎片”。

**影响**

多人视频里选 `person>0` 会得到错的人/闪烁的框；单人视频通常没问题。

**建议**

- 短期：把 README/节点描述改成“逐帧最大人物 + 碎片 track”，`person` 改名为 `track_index` 并加说明；
- 正确实现：每帧保留所有框，按 IoU/外观关联出多条轨迹，再按 track 面积/长度排序选第 N 条；
  这需要重写 `_associate` 与 `run` 的取框逻辑（约 80 行）。

---

### P1-7 `install.py` 自检结果不可信

**位置**：`install.py:12-17`、`20-36`。

实测（默认运行）：

```
模型目录: /home/tony/zrepo/zmocap/models/zmocap   ← 这个目录不存在
  MISS exmr/exmr.safetensors
  MISS sapiens2/sapiens2_1b_pose.safetensors
  MISS body_models/smplx/SMPLX_NEUTRAL.npz
```

即使显式 `--models-dir /home/tony/zext/comfyui/models/zmocap`：

```
MISS exmr/exmr.safetensors            ← 实际 235 MB，但阈值写 1.5 GB
MISS sapiens2/sapiens2_1b_pose...     ← 路径实际是 exmr/sapiens2_1b_pose...
OK   body_models/smplx/...
```

问题：

- 默认路径依赖 `folder_paths`，普通 `python install.py` 下 import 失败，回退到
  `<repo>/models/zmocap`（不存在），README 的命令按文档执行必“全 MISS”。
- `REQUIRED` 与 `exmr_engine.SOURCES` 不一致（sapiens2 路径错；exmr 阈值 1.5 GB 远大于实际 235 MB）。
- 只检查 3 个文件，漏掉 flowhmr/sam 的 3 个文件、hamer、mano、yolo 等关键件与目录。

**修法**：直接复用 `nodes/exmr_engine.py` 的 `SOURCES` / `DIR_SOURCES`；
用 `folder_paths.models_dir` 或把 ComfyUI 根加入 `sys.path`；
阈值改成“文件存在 + 非空/合理最小”（例如 >1 MB），不要拍脑袋写 1.5 GB。

---

### P1-8 `scripts/make_workflow.py` 与实现/文档脱节

实际生成的 12 个节点：

```
LoadVideo → GetVideoComponents → Track/Frames/Models
  → Keypoints/BodyFeatures/HandFeatures → ExMR
  → SMPLPreviewer / Export
  → VRMLoader（悬空，两个输出都没连线）
```

问题：

- 文件头 docstring 写“SMPL2SOMA → SOMA2VRM → VRMPreviewer ← VRMLoader”，实际**没有**这三个节点；
- `loader` 节点是死节点（`ruff F841: loader assigned but never used`）；
- `ZMocapSMPLPreviewer` 的 `character` 可选输入没接；
- `ZMocapExMR.fps` 作为输入端口硬塞进 `inputs` 列表，但工作流里同时给了 widget 值，
  前端可能渲染成“端口 + widget”两份，建议用 `widget: {"name":"fps"}` 的转换写法或干脆只保留接线。

**修法**：二选一——
(a) 补全 SOMA 分支和 loader→VRM Previewer 的连线，与 docstring 对齐；
(b) 删掉 loader、改 docstring，明确“当前模板只含 SMPL 预览 + 导出”。
不要保留“文档说有、图里没有”的状态。

---

### P1-9 不接角色时 VRM 预览是空场景

**位置**：`soma/binding.py:140-216`（`write_humanoid_glb`）。

输出 GLB：`meshes: 0`、`nodes: 52`、`animations: ['zmocap']`。
前端 `zmocap_viewer.js` 只做“显示/隐藏已有对象”，不会根据节点层级画骨骼线；
`Box3().setFromObject()` 对空场景返回空盒，`look()` 直接 return。

**影响**

`ZMocapVRMPreviewer` 的默认 `character` 是 `(none / built-in skeleton)`，
此时节点预览是**全黑/只有网格地面**。用户会以为节点坏了。
SMPL 预览器不受影响（它有 10475 顶点的自带网格）。

**修法**（任选）

1. 在 `write_humanoid_glb` 里加一套小棍/关节球（复用 `nodes/sticks.py`，
   需要把它的 `_Appender` 写出逻辑接进来，或改成统一走 `_Appender`）；
2. 或让前端 `setLayer("bones")` 时根据节点层级生成 `LineSegments`（纯前端，零后端改动）；
3. 或在没有角色时强制输出 SMPL 网格预览，并提示“未接角色”。

---

### P1-10 依赖完全未声明

**位置**：`requirements.txt`（全文都是注释）、`pyproject.toml`（`dependencies = []`）。

本机实际依赖（`/home/tony/zvenv/comfyui` 实测版本）：

```
numpy 2.5.2        torch 2.13.0+cu130     ultralytics 8.4.210
smplx 0.1.28       chumpy 0.70            PyTurboJPEG 2.6.1
yacs 0.1.8         trimesh 5.2.0          av 18.1.0
Pillow 12.3.0      psutil 7.2.2           safetensors 0.8.0
opencv-python 5.0.0.93（ultralytics 依赖）
```

注意：

- `chumpy` 在 NumPy 2 + Python 3.13 下**直接 import 会失败**（本机实测 `MISS chumpy: ImportError`），
  必须经过 `support/chumpy_compat.py`；这一点必须写进安装说明。
- `psutil` 是 `support/prefetch.py` 的可选依赖；缺了只是不预取，不报错。
- `torch` 应由 ComfyUI 提供，不要在 requirements 里钉版本（会和 ComfyUI 冲突）。

**修法**：写一份可用的 `requirements.txt`（不含 torch），并在 README 说明
“装进 ComfyUI 自己的环境”；`pyproject.toml` 的 `[project] dependencies` 同步；
`[tool.setuptools] packages = ["nodes"]` 只打包了 `nodes`，`soma/support` 不会进 wheel，
如果打算走 `pip install` 必须补全（或删掉 packages 让 setuptools 自动发现）。

---

### P2-1 `strict=True` 被静默吞掉

**位置**：`nodes/retarget.py:727-733`。

```python
if strict:
    raise RuntimeError(...)       # 在 try 里
...
except Exception as exc:
    print(f"[zmocap] 载入角色失败（{exc}），回退到自带骨架")
```

实测：给一个只有 `foo_root/foo_child` 的假 GLB，`build_glb(..., strict=True)` 返回
`{'mode': 'mannequin', ...}`，没有抛错。

**影响**：预览器传了 `strict=bool(target)`，但角色骨骼名不匹配时用户只会看到“自带骨架”，
不知道角色没绑上；`ZMocapVRMLoader` 只在 mapping 为空时打印，部分匹配也静默。

**修法**：`except` 里 `if strict: raise`，或把 `raise` 移到 `try` 外；
`bind` 的 `unmapped_count` 在 >0 时打印警告。

---

### P2-2 临时帧目录泄漏

**位置**：`nodes/frames_node.py:39-49`、`:81-84`。

每次 `ZMocapFrames` 在 `/tmp` 建目录并登记到 `_KEEP`，但 `cleanup()` **没有任何调用者**
（`grep -rn 'cleanup(' nodes` 只找到定义）。

实测：`/tmp/zmocap-frames-*` 10 个目录，合计 **143 MB**；`_KEEP` 只增不减。

**建议**：帧目录写到 ComfyUI 的 temp 目录，并在一次 prompt 结束后清理；
可以挂到 `PromptServer` 的 `execution_finished`/`on_prompt_end`，或用 `atexit` + 定期 GC；
至少不要把 `/tmp` 当永久缓存。

---

### P2-3 前端：默认角色值判断错、上传文件不进下拉、模型目录角色 404

**位置**：`web/js/zmocap.js:111-125`、`:127-163`；`nodes/character.py:39-52`。

1. **默认值判断错**：`character_choices()` 的第一项是 `"(none / built-in skeleton)"`，
   但 `loadCharacter()` 判断的是 `value === "none"`。于是新建/加载节点时会对这个字符串
   发 `/view?filename=(none / built-in skeleton)&type=input` → 404 → 提示“载入失败”。
2. **上传目录**：上传按钮带 `subfolder: "zmocap"`，文件落到 `input/zmocap/`；
   而 `character_choices()` 的 `_scan(input_dir())` 只 `iterdir()` 扫一层文件，
   不扫子目录。重启后上传的角色**不会出现在下拉框**（当前 workflow 里保存的值仍能解析，但新节点选不到）。
3. **模型目录角色**：下拉框包含 `models/zmocap/characters/*`，但前端统一用
   `inputFileUrl()`（`type=input`）。只放在 `models/characters` 的角色 → `/view` 404。
   本机的两个角色恰好在 `input/` 里也各有一份，所以现在看不出这个问题；
   新丢进模型目录的角色会踩到。`ZMocapVRMLoader` 其实已经复制了一份到 output
   并发 `ui.zmocap_character`，但 `zmocap.js` 没有 `onExecuted` 处理这个 payload，复制是白做的。
4. **`.gltf`**：`CHARACTER_EXTS` 和上传 accept 都包含 `.gltf`，但 `_read_glb()` 只认 GLB 容器，
   选 `.gltf` 直接 `ValueError: 不是 GLB 文件`。

**建议**：统一“无角色”常量并与后端共享；上传后把相对路径加进 `combo.options.values` 并持久化；
`onExecuted` 用 `outputFileUrl(payload.preview)` 载入；扫描 `input/zmocap`；
要么支持 JSON glTF，要么从扩展名/accept 里去掉 `.gltf`。

---

### P2-4 `_model_path()` 的 `folder_paths` 用法会把候选全跳过

**位置**：`nodes/smpl_mesh.py:19-40`、`nodes/loader.py:26`。

```python
for root in folder_paths.get_folder_paths("zmocap") + folder_paths.get_folder_paths("zmocap_body_models"):
```

`folder_paths.get_folder_paths("zmocap")` 会 `KeyError: 'zmocap'`（实测），
异常被外层 `except` 捕获，**连第二个候选都没试**；最后只能落到硬编码的
`Path.home()/"zext/comfyui/models/zmocap/..."`。

**影响**：换一台机器/换用户，SMPL 预览器找不到 `SMPLX_NEUTRAL.npz`；而模型明明在
`ComfyUI/models/zmocap/body_models/smplx/`。

**修法**：复用 `nodes.loader.models_root()`，不要自己拼 `get_folder_paths`；
`candidate_roots()` 里的 `Path.home()/"zext/..."` 也应删掉或改成可配置。

---

### P2-5 标准 humanoid 的 rest 基有两处解剖学错误

**位置**：`soma/humanoid.py:58-69`、`:38-46`。

- `bone_directions()` 对 hips 取“最长子骨”：hips 的子骨里 `leftUpperLeg` 略长于 `spine`，
  于是 hips 的 Y 轴指向腿（几乎水平），而不是向上。实测 `B_hips_y = [0.98, -0.22, 0]`。
- 叶子骨（toes、手指末节）没有子骨 → 默认 `+Y`；但脚趾实际朝前（-Z），手指末节沿手指方向。
  实测 `B_toes_y = [0,1,0]`，而源方向是 `[0,-0.53,-0.85]`。

**影响**：修 P0-2 之后，这些错误在绑定里会被抵消（`B_can` 在 `W = A A_rest⁻¹ B_char` 中消掉），
但 `write_humanoid_glb`（标准骨架预览）和 `soma_to_vrm` 的 Z 轴参考仍会用到它们。
如果保留“无角色标准骨架”这条展示路径，建议修：
hips 显式用 `spine` 子骨；toes 用 `(0,-0.05,-0.08)` 的解剖方向；手指末节继承父骨方向。

另：`:38-39` 有一个 `for finger, off in (...): pass` 的空循环，删掉。

---

### P2-6 glTF 规范细节：SCALAR accessor 的 `min` 是标量而不是数组

**位置**：`nodes/retarget.py:230-232`。

`times` 是 1D，`array.min(axis=0)` 返回标量，`accessor["min"] = 0.0`。
glTF 2.0 要求 `min/max` 是与 `type` 分量数一致的数组（SCALAR → 长度 1 的数组）。
实测 SMPL GLB 里有 1 个这样的 accessor（`type=SCALAR, min=0.0`）。
three.js 不校验所以能显示，但 glTF Validator/其他引擎会告警。

**修法**：`np.atleast_1d(array.min(axis=0)).tolist()`。

---

### P2-7 许可与合规

- `vendor/ExMR/LICENSE` 存在；`vendor/motionbridge` **没有任何 LICENSE/NOTICE**，
  但 `soma/vrm.py` 注释声称 Apache-2.0；`NOTICE` 只提了 ExMR/Kimodo/three.js。
- `vendor/KimodoMotionCorrection`（908 KB）从未被 import，属于死重量。
- SMPL-X 需要 Max Planck 的许可证；README 只写“需要 license”，没有给获取链接/放置说明。
- 如果把 vendor 提交进仓库，必须同时提交各上游 LICENSE 与修改说明（Apache-2.0 要求）。

---

### P2-8 没有 CI、没有 pytest、单测覆盖不到关键路径

- 仓库没有 `.github/`，没有 `pytest.ini`/`pyproject` 的 test 配置。
- 唯一测试 `tools/test_retarget.py` 是手写脚本（`python -m ...`），只测 SMPL 链的 6 骨链；
  没有测 SOMA/VRM、绑定、导出、名称映射、静止不变式。
- 它对 `_quat_from_mat` 的分支 bug 不敏感（测试旋转都落在覆盖分支里）。

**建议**：加
1. `test_quat_roundtrip`（随机 20k，0 失败）；
2. `test_rest_invariant`（零姿态 → 输出 == 角色 rest，<0.01°）；
3. `test_hand_fk`（GLB vs `smplx` 前向，<0.01 mm）；
4. `test_pinky_not_static`；
5. `test_bvh_wellformed`（通道数 == 每帧数值数）；
6. `test_mixamo_names`（20/20）；
7. CI 里跑 `ruff check --select F,E9` + `pytest`（不需要 GPU 的纯数学测试）。

---

### P2-9 性能与显存

- `build_assets()` 每次调用约 **427 ms**（扫描 486 个 vendor `.py` 文件并改写 import），
  keypoints/body/hands/exmr 四个节点各调一次 → 每跑一遍工作流约 **1.7 秒**纯开销。
- `nodes/exmr.py::_joints()` 每次都 `smplx.SMPLX(...)` 新建模型（CPU，约几百 MB），没有缓存。
- 没有走 `comfy.model_management.load_models_gpu`：模型用 `.cuda()` 直接上卡，
  靠 `del` + `torch.cuda.empty_cache()` 回收。8 GB 卡上 Sapiens2 1B（6 GB）+ SAM/HaMeR 能跑，
  但 ComfyUI 不知道这些模型的存在，OOM 时没有自动 offload/重试。DESIGN 里“显存归 ComfyUI 管”
  目前只做到了 `free_memory`/`soft_empty_cache` 两下。
- `exmr_engine.infer()` 里 `.cuda()` 硬编码，多卡/CPU/MPS 不可用。
- 两个进程级副作用：`build_assets()` 把 `vendor`、`sapiens2`、`hamer`、`support` 插到 `sys.path[0]`
  （可能和别的自定义节点重名/遮蔽），并替换全局 `torch.hub.load`（只针对 dinov3）。
  目前没出问题，但属于“静默影响其他节点”的风险。

- **快速修法**：给 `build_assets` 加模块级 `_PATCHED` 缓存；给 `_SMPLX` 类加 `lru_cache` 类似的单例；
  模型加载走 `mm.load_models_gpu` 或至少用 `mm.get_torch_device()`。

---

### P2-10 文档与实现漂移（会误导接手人）

| 文档 | 说法 | 实际 |
|---|---|---|
| README | “视频解码用 VideoHelperSuite” | 自带 `ZMocapLoadVideo`（PyAV） |
| README | “多人物里选第 N 个” | 只跟每帧最大框（P1-6） |
| README | BVH ✅ | BVH 非法（P0-3） |
| README | 节点表 | 缺 SMPL2SOMA/SOMA2VRM/两个 Previewer/VRM Loader；`Rig` 已不存在 |
| README | `[RGBMocap](https://github.com/)` | 链接是占位符 |
| DESIGN | `→ Rig ← VRM Loader` | `ZMocapRig` 未注册；绑定在 Previewer 里 |
| DESIGN | “模型经 comfy.model_management 装载” | 实际 `.cuda()`（P2-9） |
| DESIGN | “YOLOX+BoT-SORT+OSNet” | 实际 YOLOv8 + 贪心 IoU |
| DESIGN | 足部锁定/里程碑 | 未实现 |
| RESEARCH | “SOMA 77 关节” | 实现是 78 |
| `nodes/__init__.py` docstring | “Rig 只负责动作+角色→GLB” | 没有 Rig 节点 |

---

### P3 代码卫生（ruff 结果与死代码）

`ruff check --select F,E9 --exclude vendor`：

```
nodes/export.py:61        F541 f-string without placeholders
nodes/retarget.py:133     F401 torch imported but unused
nodes/retarget.py:444     F841 rest_world assigned but never used
nodes/sticks.py:90        F841 cache assigned but never used
scripts/make_workflow.py:106 F841 loader assigned but never used
soma/binding.py:142/143   F401 json, struct imported but unused
soma/binding.py:172       F841 pindex assigned but never used
soma/vrm.py:15            F401 vendor.motionbridge.retarget imported but unused
soma/vrm.py:20            F401 _quat_mul imported but unused
tools/fk_check.py:237     F841 parents assigned but never used
tools/fk_check.py:242     F841 smplx_parent assigned but never used
```

其它死代码/重复：

- `nodes/retarget.py`：`SMPLX_PARENTS` 定义了两次（22 行与 554 行，内容相同）；
  `_parent_frame()` 在 `return rotation.T, scale` 后还有一句不可达的 `return True`；
  `ZMocapRig`（743–829 行，约 87 行）未注册、与两个 Previewer 重复。
- `soma/vrm.py`：`_axis_angle` 定义了两次（149 与 165 行）；`retarget_to_rig`（176 行起）、
  `_world_matrix`、`_quat_to_xyzw`、`_quat_to_gltf`、`_Z_TO_Y` 均无人调用。
- `tools/fk_check.py`：`_quat_from_mat` 无人调用，且复制了 P1-4 的 bug。
- `nodes/retarget.py::_add_animation` 里 `rest_world = _rest_world(doc)` 算完不用。
- `nodes/loader.py`：`models["low_memory"]` 无人读取（各阶段有自己的开关）。
- `nodes/exmr.py`：`with_hands` 分支永远不会是 False（`hand_features` 是 required 输入）。
- `nodes/character.py`：日志 `52/21 humanoid bones` 的分母 21 是错的（应为 55 或 52）。
- `retarget/` 目录为空（重构残留）。
- `nodes/__pycache__/preview.cpython-313.pyc` 对应的 `preview.py` 已不存在。

**建议**：接手第一周做一次 `ruff --fix` + 删除死代码，但**先修 P0/P1 再加测试**，
避免大 diff 淹没行为变化。

---

### P3 其它小问题

- `nodes/export.py`/`previewer.py` 的 `filename` 直接拼路径，`filename="../../x"` 可以写出 output 目录之外；
  用 `Path(filename).name` 清洗。
- `ZMocapSMPLPreviewer` 的 digest 只含 `[smplx, frames, fps]`，不含 betas/内容；
  同帧数同 fps 的不同动作会复用同一个文件名（前端有 `&t=` 缓存破坏，但磁盘文件会被覆盖）。
  建议把 `betas` 与 `joints` 的哈希加进去。
- `nodes/frames.py`：RGBA 输入会 `OSError: cannot write mode RGBA as JPEG`（实测）；
  ComfyUI 主流是 RGB，但上游节点给 RGBA 时会炸。先 `frame[..., :3]` 或在 C==4 时存 PNG。
- `nodes/track.py:68` 无条件 `torch.cuda.empty_cache()`；CPU-only 构建下不优雅，包一层 `torch.cuda.is_available()`。
- `nodes/loader.py:26` 与 `nodes/smpl_mesh.py:35` 硬编码 `Path.home()/"zext/..."`，属于本机路径进产品代码。
- `ZMocapVRMLoader` 每次 `shutil.copyfile` 15 MB 到 output，且从不清旧文件；output 已积累多份 15 MB GLB。
- `scripts/dev-sync.sh` 硬编码 `tony@192.168.0.102`（开发脚本，可接受，但应写“仅本机开发用”）。
- `setup-dual-remote.sh` 会同时推 GitHub + Gitee；确认这是期望行为。
- `ZMocapExMR` 的 `static_conf_logits` 没有写进 NPZ，未来足部锁定/接触检测会需要它。
- `ZMocapLoadVideo` 的 `fps` 只能做整数抽帧：30→24 时 `round(30/24)=1`，实际仍是 30 fps；
  target > source 时直接忽略请求。用户以为设了 24，下游 `fps` 却是 30。加提示或做真正的时间重采样。
- `soma/vrm.py::write_animation` 把新动画 **append** 到 `doc["animations"]`；如果角色自带动画，
  官方 Preview3D/其他播放器可能只播第 0 条（本项目的 three.js 播放器会全播）。
  建议插到最前或清掉同名动画。
- `nodes/retarget.py::_add_bone_gizmos` 会额外写一条 `zmocap_bones` animation（与 `zmocap` 分开）；
  自定义播放器没问题，官方 Preview3D 可能只播第一条，导致小棍静止。
- `nodes/sticks.py::add_stick_layer` 的 `parent_map` 参数只写不读（死参数），可以删。
- `nodes/retarget.py::_add_animation` 的 `rest_world` 局部变量未使用（ruff F841）。

---

## 5. 两条绑定链的数学判定（为什么 P0-2 是 P0）

### 5.1 SMPL→角色（`nodes/retarget.py::_add_animation`）

输入是 SMPL-X 的局部增量 `R_i`（rest 时全 I）。公式：

```
D_i     = D_parent · R_i
L_i(t)  = D_parent⁻¹ · R_rest(i) · D_parent · R_i
```

`R_i=I, D=I` 时 `L_i = R_rest`，所以**零姿态严格还原 rest**。实测 0.0°，
单测 4e-6°。这条链可以信（除了 P1-4 的四元数边界和 P1-3 的命名映射）。

### 5.2 SOMA→VRM→角色（`soma/vrm_motion.py` + `soma/binding.py`）

输入是**绝对朝向** `A(t)`（由 SOMA 关节方向构造），绑定公式：

```
W_char(t) = A(t) · B_can⁻¹ · B_char
```

推导要求 `A(rest) = B_can`（标准骨架 rest）。但 `soma_to_vrm` 给出的 `A(rest)` 是
**源 rest 基**（SMPL-X 的脚/髋/手朝向），不是 `B_can`。于是：

```
W_char(rest) = B_source · B_can⁻¹ · B_char ≠ B_char
```

角色被拧到源 rest。正确写法是先把 A 归一化：

```
A_norm(t) = A(t) · A_rest⁻¹ · B_can
W_char(t) = A_norm(t) · B_can⁻¹ · B_char = A(t) · A_rest⁻¹ · B_char
```

`A_rest` 用 `motion.rest` 的关节位置 + 单位旋转，走同一套方向构造得到。
零姿态实测从 154.29° 降到 **0.000°**。

**旁证**：`nodes/retarget.py` 的 SMPL 链是“增量”输入，天然满足 `A(rest)=I`；
SOMA 链是“绝对”输入，少了“减掉源 rest”这一步。这不是符号/转置问题，
而是**运动表示语义**问题，修的时候不要只改一个矩阵乘法。

### 5.3 四元数分支 bug 为什么“偶尔”才看到

`_quat_from_mat` 只在 `trace<=0` 且 `m00>m11` 但 `m22>=m00` 时不命中任何分支。
真实动作的局部旋转大多是小角度（trace>0），所以 107 帧里只命中 1 次；
但随机旋转 10.4% 会命中，手指/脚这种快速小关节更容易踩到。
表现是“某一帧某一根骨突然回到 rest/抽一下”，很难肉眼定位。修它成本极低（20 行），
建议和 P0-2 一起做。

---

## 6. 修复顺序与最小补丁建议

> 下面的补丁只是建议，未写入代码。每步都给出验收命令，建议一步一提交。

### Step 0：重启服务，建立基线

当前 8188 是 17:56 的旧进程，先确认没有别的任务在跑，再重启：

```bash
cd /home/tony/zrepo/ComfyUI
# 停掉旧进程（确认 PID/命令行后再执行）
kill 27649
/home/tony/zvenv/comfyui/bin/python main.py --listen 0.0.0.0 --port 8188 --enable-manager
```

基线测试（不需要 GPU）：

```bash
cd /home/tony/zrepo
/home/tony/zvenv/comfyui/bin/python -m zmocap.tools.test_retarget
/home/tony/zvenv/comfyui/bin/python -m ruff check --select F,E9 --exclude vendor zmocap
```

### Step 1：修 `_quat_from_mat`（P1-4）

`nodes/retarget.py:301`，分支改成互斥穷尽（见 §4 P1-4）；
`tools/fk_check.py:86` 同步（或抽公共模块）。
加 `test_quat_roundtrip`。

验收：随机 20k 旋转 round-trip 0 失败。

### Step 2：修 SOMA→VRM 的 rest 归一化（P0-2）

`soma/vrm_motion.py`：抽 `_orientations()`，对 rest 再跑一次，按 §5.2 归一化。
加 `test_rest_invariant`（零姿态输出 == 角色 rest，<0.01°）。

验收：§3.4 的 42/52 变成 0/52；真实动作的 `W(t)W(0)ᵀ` vs `A(t)A(0)ᵀ` < 0.01°。

### Step 3：修 `flat_hand_mean`（P1-1）

`nodes/exmr.py:96` 改 `True`。验收：手部 FK 最大误差 < 0.01 mm。

### Step 4：修 Pinky 与 Mixamo 映射（P1-2 / P1-3）

`soma/vrm_motion.py` 的 `_BONES`/`_LEAF_FALLBACK`；`soma/binding.py` 的 `_ALIAS` 补全。
验收：6 根 Little 骨有动画；20 个 Mixamo 名称 20/20；真实 Mixamo GLB 绑出 52 骨。

### Step 5：修绑定 scale（P1-5）

`_leg_length` + `matched_heads` 的键。验收：两个身高差 20% 的角色 `scale` 不同。

### Step 6：修导出（P0-3）

按 `skeleton` 分支；BVH 写满通道或改成逐关节 position 通道。
加 `test_bvh_wellformed`。验收：Blender/BVH viewer 能打开，帧数/通道数一致。

### Step 7：修 install / workflow / vendor / requirements（P0-1、P1-7、P1-8、P1-10）

- `install.py` 复用 `exmr_engine.SOURCES`；
- `make_workflow.py` 补 SOMA 分支或删 loader；
- vendor 三选一（提交/submodule/fetch 脚本）+ 许可；
- requirements 写实。

验收：全新 clone + `pip install -r requirements.txt` + `python install.py --models-dir ...` 全绿；
`git archive` 出来的目录能 import `soma.binding`。

### Step 8：UX 与卫生（P1-9、P2-1 … P3）

- VRM 无角色时加小棍/前端骨骼线；
- `strict` 不再吞错；
- 临时帧清理；
- 前端默认值/上传/模型目录；
- `_model_path` 用 `models_root()`；
- `.gltf` 明确不支持；
- 文档同步；
- 死代码清理。

---

## 7. 验收清单（Definition of Done）

发布/交接前至少满足：

- [ ] 全新 clone（不含本机 vendor/models）能 `import zmocap.soma.binding`，能跑 ExMR（vendor 就位后）。
- [ ] `requirements.txt` 非空，按文档安装后 `ultralytics/smplx/chumpy/trimesh/av/PIL` 可导入；
      chumpy 的兼容说明在 README。
- [ ] `test_quat_roundtrip`：20k 随机旋转 0 失败。
- [ ] `test_rest_invariant`：零姿态 SMPL → SOMA → VRM → 角色，逐骨 < 0.01°。
- [ ] `test_hand_fk`：GLB 手部关节 vs `smplx` 前向 < 0.01 mm。
- [ ] `test_pinky_not_static`：6 根 Little 骨方差 > 0。
- [ ] `test_mixamo_names`：20/20。
- [ ] `test_bvh_wellformed`：通道数 == 每帧数值数，帧数 == Frames。
- [ ] `ZMocapExport` 对 `smplx/soma/vrm` 三种动作都不 KeyError（或给出明确错误）。
- [ ] `install.py` 在真实模型目录上全 OK；默认命令不再指向不存在的路径。
- [ ] `make_workflow.py` 生成的图没有悬空节点；docstring 与图一致。
- [ ] VRM Previewer 不接角色时能看见骨架（不是空场景）。
- [ ] 重启服务后 `/object_info` 的 16 个节点与磁盘代码一致。
- [ ] `ruff check --select F,E9` 干净。
- [ ] LICENSE/NOTICE 覆盖 vendor 的 ExMR + MotionBridge + three.js。

---

## 8. 风险登记与待决问题

| 风险 | 影响 | 建议 |
|---|---|---|
| vendor 不在仓库 | 新用户/CI 不可用；许可无法合规 | P0-1，优先决策提交还是 submodule |
| SOMA 链静止畸变 | 角色预览/导出整体姿势错 | P0-2，优先修 |
| 8 GB 显存 + 6 GB Sapiens2 | OOM/速度；ComfyUI 不知道这些模型 | 考虑 fp16、分块、`load_models_gpu` |
| 上游 ExMR 版本漂移 | vendor 重新拉取后接口变动 | 固定 commit + sha256 + smoke test |
| 前端未实测 | 默认值/上传/缓存类 bug 只在浏览器出现 | 用 Playwright 或人工过一遍 |
| 模型许可（SMPL-X） | 不能随仓库分发 | README 写清获取方式，不要提交模型 |
| MotionBridge 许可缺失 | 若 vendor 进仓库会违规 | 补 LICENSE/NOTICE 或删除死代码依赖 |
| `ZMocapRig` 是否保留 | 与两个 Previewer 重复 | 删除或注册并在文档统一 |
| `.gltf` 支持 | 用户选 .gltf 直接报错 | 明确不支持或实现 |
| 输出目录膨胀 | 15 MB GLB/次，无清理 | 加保留策略/清理 |

**待决**：

1. 发布范围：只发 SMPL 链（数学闭环）还是连同 SOMA 链一起发？如果近期发，必须先修 P0-2。
2. vendor 策略：(a) 提交 12 MB、(b) submodule、(c) fetch 脚本。
3. 是否保留 `ZMocapRig` / `retarget_to_rig` / MotionBridge 依赖。
4. 多人跟踪是“近期必须”还是“以后再说”；如果以后，先改 README。
5. 是否要支持 Mixamo 角色（当前 SOMA 链不支持，SMPL 链支持）——这会影响 `_ALIAS` 的工作量。

---

## 9. 复现命令与关键路径

### 9.1 环境

```bash
PY=/home/tony/zvenv/comfyui/bin/python
cd /home/tony/zrepo
$PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"
$PY -c "import folder_paths; print(folder_paths.models_dir)"   # 需要 ComfyUI 在 sys.path
```

### 9.2 单测与静态检查

```bash
cd /home/tony/zrepo
$PY -m zmocap.tools.test_retarget
$PY -m ruff check --select F,E9 --exclude vendor zmocap
$PY -m compileall -q zmocap
```

### 9.3 静止不变式（P0-2 的回归测试原型）

```python
import sys, numpy as np, torch
sys.path.insert(0, "/home/tony/zrepo")
from zmocap.soma.motion import build_soma_motion
from zmocap.soma import vrm_motion
from zmocap.soma.binding import bind
from zmocap.soma.vrm import write_animation
from zmocap.tools import fk_check

char = "/home/tony/zrepo/ComfyUI/output/zmocap_character_Sendagaya_Shino.glb"
T = 3
z = lambda w: torch.zeros(T, w)
P = {"betas": z(10), "global_orient": z(3), "body_pose": z(63),
     "left_hand_pose": z(45), "right_hand_pose": z(45), "transl": z(3)}
motion = vrm_motion.soma_to_vrm(build_soma_motion(P))
res = bind(motion, char)
out = write_animation(char, "/tmp/rest_check.glb", res, 30.0, sticks=False)

def world(doc, state=None):
    nodes = doc["nodes"]; parent = {}
    for i, n in enumerate(nodes):
        for c in n.get("children", []) or []:
            parent[c] = i
    cache = {}
    def rot(i):
        if i in cache: return cache[i]
        q = state[i][1] if state is not None else np.asarray(
            nodes[i].get("rotation") or [0, 0, 0, 1], float)
        m = fk_check.quat_to_mat(np.asarray(q, float))
        cache[i] = m if i not in parent else rot(parent[i]) @ m
        return cache[i]
    for i in range(len(nodes)): rot(i)
    return cache

orig, _ = fk_check.read_glb(char)
doc, binary = fk_check.read_glb(out)
w_rest = world(orig)
w_out = world(doc, fk_check.sample_animation(doc, binary, 1, 30.0))
index = {n.get("name"): i for i, n in enumerate(orig["nodes"])}
errs = []
for bone in res["world_rotation"]:
    i = index[bone]
    delta = w_out[i] @ w_rest[i].T
    errs.append(np.degrees(np.arccos(np.clip((np.trace(delta) - 1) / 2, -1, 1))))
print("max", round(max(errs), 3), "median", round(float(np.median(errs)), 3))
```

当前代码输出 `max 154.292 median 11.042`；修好 P0-2 后应为 `max 0.0 median 0.0`。

### 9.4 关键路径

```
仓库              /home/tony/zrepo/zmocap
ComfyUI           /home/tony/zrepo/ComfyUI
Python            /home/tony/zvenv/comfyui/bin/python
模型              /home/tony/zext/comfyui/models/zmocap
角色              .../models/zmocap/characters/*.vrm
输出              /home/tony/zrepo/ComfyUI/output/zmocap_*
日志              /home/tony/zrepo/ComfyUI/user/comfyui_8188.log
临时帧            /tmp/zmocap-frames-*
服务              http://127.0.0.1:8188
```

### 9.5 关键文件行号（本次审计时）

| 问题 | 位置 |
|---|---|
| P0-2 rest 归一化 | `soma/vrm_motion.py:63-126` |
| P0-2 绑定公式 | `soma/binding.py:97-100` |
| P0-3 导出 KeyError | `nodes/export.py:36` |
| P0-3 BVH | `nodes/export.py:55-88` |
| P1-1 flat_hand_mean | `nodes/exmr.py:96` |
| P1-2 Pinky | `soma/vrm_motion.py:36-56` |
| P1-3 Mixamo | `soma/binding.py:22-68`、`nodes/retarget.py:117` |
| P1-4 quat | `nodes/retarget.py:301-322`、`tools/fk_check.py:86-108` |
| P1-5 scale | `soma/binding.py:71-78`、`105-108` |
| P1-6 多人 | `nodes/track.py:69-82` |
| P1-7 install | `install.py:12-36` |
| P1-8 workflow | `scripts/make_workflow.py:100-119` |
| P1-9 空场景 | `soma/binding.py:140-216` |
| P2-1 strict | `nodes/retarget.py:727-733` |
| P2-2 临时帧 | `nodes/frames_node.py:39-49`、`81-84` |
| P2-3 前端 | `web/js/zmocap.js:111-163` |
| P2-4 路径 | `nodes/smpl_mesh.py:19-40` |
| P2-5 rest 基 | `soma/humanoid.py:58-69`、`38-46` |
| P2-6 accessor | `nodes/retarget.py:230-232` |

---

## 10. 一句话交接

> 这是一个**推理链已经跑通、SMPL 链数学可信、SOMA 链还差一次“源 rest 归一化”**的项目；
> 发布前先修 P0-1（vendor）、P0-2（rest 归一化）、P0-3（导出），
> 再补 P1-1/2/3/4 和回归测试。当前 8188 是旧进程，所有结论以 `ba128da` 磁盘代码为准。
