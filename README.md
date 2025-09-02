## StreamFusion CLI · 全网视频聚合搜索下载

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-2ea44f.svg)](#)

一个面向命令行的多数据源视频聚合工具：并行搜索、交互选择分组/分集、优先使用 ffmpeg 下载 m3u8，内置简易下载器兜底。

---

### 目录

- 概览
- 功能特性
- 安装与环境
- 快速开始
- 配置说明（`config.json`）
- 使用示例
- 交互与输出
- 常见问题（FAQ）
- 路线图
- 免责声明

---

### 概览

StreamFusion CLI 基于 Python，聚合 AppleCMS 生态数据源进行检索与下载。项目目标是：开箱即用、稳定、对终端用户友好。

### 功能特性

- 并行聚合搜索：同时向多个站点发起检索，返回统一候选并标注来源站点 key
- 交互式选择：先选分组（播放器/清晰度），再选分集；支持一键整组下载
- m3u8 下载策略：
  - 优先调用本机 `ffmpeg`（透传 UA/Referer）
  - 失败自动回退到内置简易下载器（显示片段进度/累计大小/实时速度）
- 智能请求头：统一浏览器 UA，Referer 优先取站点 `detail`，否则回退为 m3u8 域名
- 便捷输出：
  - 未指定 `-o`：默认保存到 `./downloads/片名[–分集].mp4`
  - 批量整组下载在 `-o` 基础上自动追加 `-01`、`-02` …
- 精简依赖：仅依赖 `requests`、`colorama`，可选 `ffmpeg`

### 安装与环境

建议使用虚拟环境：

```bash
# Windows PowerShell
python -m venv venv
./venv/Scripts/Activate.ps1

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

可选（推荐）：安装 ffmpeg，以获得更稳健的合并与解封装能力。

### 配置说明（`config.json`）

在项目根目录放置 `config.json`。核心字段为 `api_site`，每个站点至少包含：

```json
{
  "api_site": {
    "dyttzy": {
      "api": "http://caiji.dyttzyapi.com/api.php/provide/vod",
      "name": "电影天堂资源",
      "detail": "http://caiji.dyttzyapi.com"
    }
    
  }
}
```

- `api`：AppleCMS V10 兼容接口根地址（例如 `.../api.php/provide/vod`）
- `detail`：可选，用作 Referer 构建；缺省时回退至 m3u8 域名

说明：工具调用各站点 AppleCMS 兼容接口进行搜索（`?ac=list&wd=关键词`）与详情（`?ac=detail&ids=...`），并从返回数据解析 `vod_play_url`/`vod_play_url_multi` 获取 m3u8。

### 快速开始

```bash
# 全站聚合搜索并下载（交互式）
python streamfusion_cli.py -q "三体" -v

# 仅在指定站点搜索
python streamfusion_cli.py -q "三体" -s dyttzy -v

# 指定输出文件名（单集）
python streamfusion_cli.py -q "三体" -o out/videos/三体.mp4 -v
```

行为约定：

- 未指定 `-o`：单集保存为 `./downloads/片名.mp4`；多集保存为 `./downloads/片名-第XX集.mp4`
- 指定了 `-o` 且选择整组下载：将自动追加序号后缀 `-01.mp4`、`-02.mp4` …

所有可用参数：

```text
-q, --query     搜索关键字（必填）
-o, --output    输出文件名（可选）
-c, --config    配置文件路径，默认 ./config.json
-v, --verbose   打印详细过程
-s, --site      指定站点 key（来源于 config.json 的 api_site）
```

### 使用示例

1) 聚合搜索并选择来源站点 → 选择分组 → 选择分集/整组：

```bash
python streamfusion_cli.py -q "以法之名" -v
```

2) 在指定站点内搜索并下载第一条：

```bash
python streamfusion_cli.py -q "初吻" -s dyttzy -v
```

3) 指定输出路径并整组下载：

```bash
python streamfusion_cli.py -q "某剧名" -o out/目标.mp4 -v
```

### 交互与输出

- 候选列表：展示名称、年份、类型、清晰度/版本备注，以及来源站点 key
- 分组/分集：先选分组（不同播放器/清晰度），再选分集；可选择 `A` 一键整组下载
- 进度显示：
  - ffmpeg：透传原生日志
  - 内置下载器：片段数、百分比、累计 MB、实时速度

### 常见问题（FAQ）

- 403 Forbidden？
  - 已默认设置浏览器 UA 并自动带 Referer。若仍失败，请在站点配置中补充 `detail` 域名，或更换站点。
  - 推荐安装 `ffmpeg` 提升兼容性。
- 搜不到结果？
  - 在 `config.json` 的 `api_site` 中添加更多站点。
  - 换用更简短的关键词或别名。
- 下载速度慢？
  - 更换站点，或在交互中选用不同分组（清晰度/线路）。
