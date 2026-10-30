# WDP-nodes

ComfyUI 自定义节点包：**上传自己的黑白遮罩图，直接转成 `MASK`**。

- 任意形状，不再受 ComfyUI-ppm `LatentToMaskBB` 的矩形限制。
- 不依赖透明通道（ComfyUI 自带的 `Load Image` 只从 alpha 生成遮罩，普通黑白 PNG 会得到一张 64×64 全零遮罩）。
- 纯白 = 选中/重绘（`1.0`），纯黑 = 保留（`0.0`），与 ComfyUI 全部遮罩节点一致。
- 反转、二值化（阈值）、拉伸/等比缩放、跟随参考图/latent 尺寸，常用动作都在一个节点里。

## 节点

| 节点 | 类别 | 说明 |
| --- | --- | --- |
| **WDP 遮罩图上传 (Load Mask Image)** | `WDP-nodes/遮罩` | 上传 PNG/JPG/WebP 遮罩图，输出 `mask` 与 `preview` |

### 输入

| 参数 | 默认 | 说明 |
| --- | --- | --- |
| `image` | — | 从 `ComfyUI/input` 选择，或点节点上的 **choose file to upload** 上传。 |
| `channel` | `luminance` | 取哪一路灰度：`luminance`（推荐）/ `red` / `green` / `blue` / `alpha` / `average` / `max`。 |
| `invert` | 关 | 白底黑图（黑=选中）时打开。 |
| `binarize` | 关 | 按 `threshold` 把遮罩压成纯 0/1，消除 JPG 噪点与抗锯齿灰边。 |
| `threshold` | `0.5` | 仅 `binarize` 开启时生效，`>=` 阈值算白色。 |
| `resize_mode` | `stretch` | `stretch` 拉伸到目标尺寸；`fit` 等比缩放 + 居中补黑边；`none` 保持原图尺寸。 |
| `width` / `height` | `0` | 手动的目标尺寸（没有连接 `reference_image` / `latent` 时生效），`0` = 原图。 |
| `reference_image`（可选） | — | 连上要重绘的 `IMAGE`，遮罩自动缩放到该图尺寸（优先级最高）。 |
| `latent`（可选） | — | 连上当前 `LATENT`，遮罩自动缩放到 latent 尺寸 ×8（像素空间），用于直接替换 `LatentToMaskBB`。 |

**目标尺寸优先级**：`reference_image` → `latent` → `width`/`height`。三者都没有时保持遮罩原尺寸；`resize_mode = none` 时一律不缩放。二值化在缩放**之后**执行，边缘更自然。

### 输出

- `mask`：`[B, H, W]` float32，范围 `0..1`。
- `preview`：`[B, H, W, 3]` 灰度预览图，方便直接看遮罩结果（可接 `Preview Image`）。

## 安装

仓库地址：**https://github.com/harry334421/WDP-nodes**

### 方式一：绘世启动器（另一个台机器最省事）

1. 启动器 → **版本管理 → 安装新扩展**；
2. 粘贴 `https://github.com/harry334421/WDP-nodes.git`（或直接贴仓库主页地址）；
3. 安装完成后重启 ComfyUI。

> 启动器拉 GitHub 只认 Windows 系统代理（`netsh winhttp`），不走它自己面板里的代理框；拉不动时先在系统里挂上 Clash 的系统代理。

### 方式二：ComfyUI-Manager

Manager → **Install Custom Nodes** → 搜索框下面切到 **Install via Git URL**，填同一个地址。

### 方式三：手动 / git

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/harry334421/WDP-nodes.git
```

或者直接把整个 `WDP-nodes` 文件夹拷进 `ComfyUI/custom_nodes/`（不需要 .git）。

装好后**重启 ComfyUI**，节点出现在右键菜单 `WDP-nodes → 遮罩` 下，也可直接搜索 `遮罩图上传` / `WDP`。

无需安装任何依赖（只用 ComfyUI 自带的 `torch` / `numpy` / `Pillow`）。

### 更新

```bash
cd ComfyUI/custom_nodes/WDP-nodes
git pull
```

启动器里则是「版本管理 → 已安装扩展 → 更新」。更新后重启 ComfyUI。

## 直接替换 LatentToMaskBB

`LatentToMaskBB` 只吃 `LATENT`，用 `x/y/w/h` 画矩形；本节点可以原位置替换：

1. 删掉 `LatentToMaskBB`，在原处加入 **WDP 遮罩图上传**。
2. 把原来接到 `LatentToMaskBB.latent` 的线改接到本节点的 **`latent`**（可选，只用来对齐尺寸）。
3. 上传自己的遮罩图，`mask` 输出接回原来的下一位（`SetLatentNoiseMask` / `VAE Encode (for Inpainting)` / `InpaintModelConditioning` 等）。
4. 遮罩与生成尺寸不一致时，`resize_mode` 保持默认 `stretch` 即可；想保持比例就用 `fit`。

> 说明：ComfyUI 的 `MASK` 是像素空间 `[B,H,W]`，下游节点会自己把它插值到像素/latent 尺寸（`VAEEncodeForInpaint`、`InpaintModelConditioning`、采样器的 `reshape_mask`）。所以尺寸差一点通常不会报错；接上 `latent` 或 `reference_image` 是为了让遮罩和生成画面严格对齐。

## 典型用法（局部重绘）

```
Load Image ──────────────┬─> VAE Encode (for Inpainting) ─> KSampler ─> ...
                         │            ▲
WDP 遮罩图上传 (mask) ───┴────────────┘
      └─(reference_image / latent)──┘   # 可选：让遮罩自动匹配尺寸
```

1. 把画好的黑白遮罩图上传到节点。
2. 白色区域必须正好是要重绘的地方；如果是白底黑图，打开 `invert`。
3. 尺寸不一致时，把 `Load Image` 的 `IMAGE` 或上游 `LATENT` 接进来。
4. 遮罩图是 JPG 或有灰边，打开 `binarize`。

## 常见问题

- **遮罩全黑 / 全白**：先看 `invert`；再确认 `channel` 选择正确（彩色遮罩图用 `luminance`）。
- **上传后列表里没有新文件**：右键节点 → `Refresh Node`，或刷新页面；文件本体已在 `ComfyUI/input` 下。
- **遮罩尺寸和图片对不上**：接 `reference_image` 或 `latent`，或手动填 `width`/`height`。
- **画质很脏的 JPG 遮罩**：`binarize` + `threshold 0.5`，必要时调到 `0.3 ~ 0.7` 之间试。
- **多帧图片**（GIF/APNG/动态 WebP）：所有帧会作为 batch 输出；帧尺寸不一致的帧会被跳过。

## 目录结构

```
WDP-nodes/
├─ __init__.py        # 注册节点
├─ nodes_mask.py      # 节点实现
├─ README.md
├─ LICENSE            # MIT
├─ .gitattributes     # 统一 LF
└─ .gitignore
```
