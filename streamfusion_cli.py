import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
import concurrent.futures

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except Exception:  # pragma: no cover
    class Dummy:
        RESET_ALL = ""
    class ForeDummy:
        GREEN = ""
        YELLOW = ""
        RED = ""
        CYAN = ""
        MAGENTA = ""
    class StyleDummy:
        RESET_ALL = ""
    Fore = ForeDummy()  # type: ignore
    Style = StyleDummy()  # type: ignore


class Logger:
    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose

    def info(self, msg: str) -> None:
        if self.verbose:
            print(f"{Fore.GREEN}[INFO]{Style.RESET_ALL} {msg}")

    def warn(self, msg: str) -> None:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} {msg}")

    def error(self, msg: str) -> None:
        print(f"{Fore.RED}[ERROR]{Style.RESET_ALL} {msg}")


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def build_session(default_headers: Optional[Dict[str, str]] = None) -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": DEFAULT_UA,
        "Accept": "*/*",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Connection": "keep-alive",
    })
    if default_headers:
        session.headers.update(default_headers)
    return session


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_api_base(api_root: str) -> str:
    # 规范化 api 根路径，去掉末尾的斜杠
    return api_root.rstrip("/")


def test_site_speed_and_search(session: requests.Session, api_root: str, keyword: str, timeout: float = 6.0) -> Tuple[float, List[Dict[str, Any]]]:
    """在单站点进行搜索并返回耗时与结果。无结果时返回空列表。"""
    base = build_api_base(api_root)
    params = {"ac": "list", "wd": keyword}
    start = time.time()
    try:
        resp = session.get(base, params=params, timeout=timeout)
        elapsed = time.time() - start
        resp.raise_for_status()
        data = resp.json()
        # AppleCMS V10 一般为 { list: [ { id, vod_name, ... } ] }
        result = data.get("list") or data.get("data") or []
        return elapsed, result
    except Exception:
        return float("inf"), []


def pick_fastest_site_with_results(session: requests.Session, logger: Logger, api_sites: Dict[str, Dict[str, Any]], keyword: str) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """在多个站点并行择优（顺序快速实现版：顺序测速，记录最优）。"""
    best_key: Optional[str] = None
    best_site: Optional[Dict[str, Any]] = None
    best_results: List[Dict[str, Any]] = []
    best_elapsed = float("inf")

    for key, site in api_sites.items():
        api = site.get("api")
        if not api:
            continue
        logger.info(f"开始测速与搜索站点 `{key}` -> {api}，关键词：{keyword}")
        elapsed, results = test_site_speed_and_search(session, api, keyword)
        logger.info(f"站点 `{key}` 搜索耗时 {elapsed:.2f}s，返回结果数：{len(results)}")
        if results and elapsed < best_elapsed:
            best_elapsed = elapsed
            best_key = key
            best_site = site
            best_results = results

    if not best_site:
        raise RuntimeError("所有站点均无搜索结果或不可用")
    return best_key or "", best_site, best_results


def search_all_sites(session: requests.Session, logger: Logger, api_sites: Dict[str, Dict[str, Any]], keyword: str) -> List[Tuple[str, Dict[str, Any], Dict[str, Any], float]]:
    """并行遍历全部站点搜索，返回 (site_key, site_obj, result_item, elapsed) 列表。"""
    aggregated: List[Tuple[str, Dict[str, Any], Dict[str, Any], float]] = []

    def job(item: Tuple[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]], float]:
        key, site = item
        api = site.get("api")
        if not api:
            return key, site, [], float("inf")
        logger.info(f"[并行] 开始搜索 `{key}` -> {api}")
        elapsed, results = test_site_speed_and_search(session, api, keyword)
        logger.info(f"[并行] `{key}` 完成 {elapsed:.2f}s，结果数：{len(results)}")
        return key, site, results or [], elapsed

    items = list(api_sites.items())
    if not items:
        return []
    max_workers = min(len(items), 8)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(job, it) for it in items]
        for fut in concurrent.futures.as_completed(futures):
            key, site, results, elapsed = fut.result()
            for r in results:
                aggregated.append((key, site, r, elapsed))
    return aggregated


def fetch_detail_by_ids(session: requests.Session, api_root: str, ids: List[str], timeout: float = 10.0) -> Dict[str, Any]:
    base = build_api_base(api_root)
    params = {"ac": "detail", "ids": ",".join(ids)}
    resp = session.get(base, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_play_urls(vod: Dict[str, Any]) -> List[str]:
    """从 AppleCMS 详情的单条影片记录中解析 m3u8 播放地址列表。
    兼容字段：vod_play_url / play_url / vod_play_url_multi
    格式一般为：name$uri#name2$uri2，多个播放器以 $$$ 分隔。
    这里只提取所有 uri，返回优先 m3u8 的地址列表。
    """
    raw = vod.get("vod_play_url") or vod.get("play_url") or ""
    if not raw and "vod_play_url_multi" in vod:
        try:
            # 有些实现是数组
            for item in vod["vod_play_url_multi"]:
                raw = raw + ("#" if raw else "") + (item.get("url") or "")
        except Exception:
            pass

    if not raw:
        return []

    playlists: List[str] = []
    # 去除播放器拆分 $$$，逐段解析
    for block in str(raw).split("$$$"):
        for entry in block.split("#"):
            entry = entry.strip()
            if not entry:
                continue
            if "$" in entry:
                _, url = entry.rsplit("$", 1)
            else:
                url = entry
            url = url.strip()
            if url:
                playlists.append(url)

    # 优先 m3u8
    m3u8_first = [u for u in playlists if ".m3u8" in u.lower()]
    others = [u for u in playlists if u not in m3u8_first]
    return m3u8_first + others


def pick_first_m3u8(play_urls: List[str]) -> Optional[str]:
    for u in play_urls:
        if ".m3u8" in u.lower():
            return u
    return None


def parse_play_groups(vod: Dict[str, Any]) -> List[List[Tuple[str, str]]]:
    """将 vod_play_url 解析为分组与分集：[[ (title, url), ... ], ...]。
    分组为不同播放器/清晰度源，分集为同一组内的多集条目。
    """
    raw = vod.get("vod_play_url") or vod.get("play_url") or ""
    groups: List[List[Tuple[str, str]]] = []
    if not raw and "vod_play_url_multi" in vod:
        try:
            tmp_blocks: List[str] = []
            for item in vod["vod_play_url_multi"]:
                tmp_blocks.append(item.get("url") or "")
            raw = "$$$".join([b for b in tmp_blocks if b])
        except Exception:
            pass
    if not raw:
        return groups
    for block in str(raw).split("$$$"):
        entries: List[Tuple[str, str]] = []
        for entry in block.split("#"):
            ent = entry.strip()
            if not ent:
                continue
            title = ""
            url = ent
            if "$" in ent:
                parts = ent.rsplit("$", 1)
                title = parts[0].strip()
                url = parts[1].strip()
            entries.append((title or url, url))
        if entries:
            groups.append(entries)
    return groups


def choose_episode(logger: Logger, groups: List[List[Tuple[str, str]]]) -> Tuple[Optional[int], List[Tuple[str, str]]]:
    """交互选择分组与分集。
    返回 (group_index, episodes)，其中 episodes 为选择下载的 (title, url) 列表，可为多集。
    """
    if not groups:
        raise RuntimeError("未解析到任何分组/分集")
    logger.info(f"检测到 {len(groups)} 个播放源分组")
    # 选择分组
    if len(groups) > 1:
        for gi, grp in enumerate(groups, 1):
            print(f"  [G{gi}] 分组{gi}（{len(grp)} 集）")
        while True:
            selg = input(f"请选择分组 1-{len(groups)}（回车默认1）：").strip()
            if not selg:
                gidx = 1
                break
            if selg.isdigit() and 1 <= int(selg) <= len(groups):
                gidx = int(selg)
                break
            print("输入无效")
    else:
        gidx = 1
    group = groups[gidx - 1]
    # 选择分集
    limit = min(len(group), 500)
    for i, (title, _url) in enumerate(group[:limit], 1):
        print(f"  [{i}] {title}")
    print("  [A] 下载整组全部剧集")
    while True:
        sel = input(f"请选择分集 1-{limit} 或输入 A 全部下载：").strip().lower()
        if sel == "a":
            return gidx, group[:limit]
        if sel.isdigit() and 1 <= int(sel) <= limit:
            epi = int(sel)
            return gidx, [group[epi - 1]]
        print("输入无效")


def has_ffmpeg() -> bool:
    from shutil import which
    return which("ffmpeg") is not None


def download_via_ffmpeg(m3u8_url: str, output: str, headers: Optional[Dict[str, str]] = None) -> int:
    import subprocess
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "info",
        "-y",
    ]
    if headers:
        # ffmpeg 通过 -headers 传递，使用 CRLF 分隔
        header_lines = "".join([f"{k}: {v}\r\n" for k, v in headers.items()])
        cmd += ["-headers", header_lines]
    cmd += [
        "-i", m3u8_url,
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",
        output,
    ]
    # 以流式方式输出进度（stderr）
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, text=True)
    if proc.stderr is not None:
        for line in proc.stderr:
            line = line.rstrip()
            if not line:
                continue
            # 直接透传 ffmpeg 日志作为进度参考
            print(line)
    proc.wait()
    return int(proc.returncode or 0)


def simple_m3u8_download(session: requests.Session, m3u8_url: str, output: str, timeout: float = 10.0) -> None:
    """极简 m3u8 下载：
    1) 拉取 m3u8 内容，如为主播放清单（含 EXT-X-STREAM-INF），挑选第一个子清单
    2) 拉取子清单，顺序下载 ts 片段并追加写入
    仅适用于简单场景；如需强大功能建议安装 ffmpeg。
    """
    from urllib.parse import urljoin

    def get_text(url: str) -> str:
        r = session.get(url, timeout=timeout)
        r.raise_for_status()
        r.encoding = "utf-8"
        return r.text

    text = get_text(m3u8_url)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 若为主清单，选第一条子清单（或带最高带宽的）
    if any(ln.startswith("#EXT-X-STREAM-INF") for ln in lines):
        child_url: Optional[str] = None
        last_bw = -1
        for i, ln in enumerate(lines):
            if ln.startswith("#EXT-X-STREAM-INF"):
                bw = last_bw
                try:
                    # 简单提取 BANDWIDTH
                    for part in ln.split(","):
                        if part.strip().startswith("BANDWIDTH="):
                            bw = int(part.split("=", 1)[1])
                            break
                except Exception:
                    bw = -1
                # 下一行为子清单 URL
                if i + 1 < len(lines):
                    cand = urljoin(m3u8_url, lines[i + 1])
                    # 挑最大带宽
                    if bw >= last_bw:
                        child_url = cand
                        last_bw = bw
        if child_url:
            m3u8_url = child_url
            text = get_text(m3u8_url)
            lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # 收集片段 URL
    seg_urls: List[str] = []
    for ln in lines:
        if ln.startswith("#"):
            continue
        seg_urls.append(requests.compat.urljoin(m3u8_url, ln))

    # 逐段下载并写入 .ts，再尝试转换为 .mp4（简单重命名不安全，这里只输出 .ts）
    tmp_ts = output if output.lower().endswith(".ts") else output + ".ts"
    total_segments = len(seg_urls)
    bytes_downloaded = 0
    start_time = time.time()

    def print_progress(seg_idx: int, seg_total: int, bytes_dl: int) -> None:
        elapsed = max(time.time() - start_time, 1e-6)
        speed = bytes_dl / 1024 / 1024 / elapsed  # MB/s
        percent = (seg_idx / seg_total * 100.0) if seg_total else 0.0
        sys.stdout.write(f"\r片段进度: {seg_idx}/{seg_total}  总计: {bytes_dl/1024/1024:.2f}MB  速度: {speed:.2f}MB/s  完成: {percent:.1f}%")
        sys.stdout.flush()

    with open(tmp_ts, "wb") as f:
        for idx, url in enumerate(seg_urls, 1):
            r = session.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)
                    bytes_downloaded += len(chunk)
                    print_progress(idx - 1 + 0.5, total_segments, bytes_downloaded)
            print_progress(idx, total_segments, bytes_downloaded)
    print()  # 换行

    # 若用户目标是 .mp4 且本机有 ffmpeg，尝试封装
    if output.lower().endswith(".mp4") and has_ffmpeg():
        code = download_via_ffmpeg(m3u8_url, output)
        if code == 0:
            try:
                os.remove(tmp_ts)
            except Exception:
                pass


def choose_from_results(logger: Logger, results: List[Dict[str, Any]], with_site: bool = False) -> Tuple[Dict[str, Any], Optional[str], Optional[Dict[str, Any]]]:
    # 展示前 30 条供选择
    limit = min(len(results), 30)
    logger.info(f"共 {len(results)} 条结果，展示前 {limit} 条：")
    for i, entry in enumerate(results[:limit], 1):
        if with_site:
            item = entry[2]  # type: ignore[index]
            site_key = entry[0]  # type: ignore[index]
        else:
            item = entry  # type: ignore[assignment]
            site_key = None
        name = item.get("vod_name") or item.get("name") or "未知"
        year = item.get("vod_year") or item.get("year") or "?"
        type_name = item.get("type_name") or item.get("vod_class") or ""
        # 清晰度/版本/备注等尽可能展示
        remark = (
            item.get("vod_remarks")
            or item.get("vod_note")
            or item.get("vod_version")
            or item.get("remarks")
            or item.get("note")
            or item.get("version")
            or ""
        )
        remark_str = f" [{remark}]" if remark else ""
        extra = f" 站点:{site_key}" if with_site else ""
        print(f"  [{i}] {name}{remark_str} ({year}) {type_name}{extra}")
    while True:
        sel = input(f"请选择序号 1-{limit}：").strip()
        if not sel.isdigit():
            print("请输入数字序号")
            continue
        idx = int(sel)
        if 1 <= idx <= limit:
            if with_site:
                chosen = results[idx - 1]
                return chosen[2], chosen[0], chosen[1]  # type: ignore[index]
            return results[idx - 1], None, None  # type: ignore[return-value]
        print("序号超出范围")


def run(output: Optional[str], config_path: str, verbose: bool, site_key_opt: Optional[str] = None, query: Optional[str] = None) -> None:
    logger = Logger(verbose=verbose)
    logger.info("读取配置文件…")
    config = load_config(config_path)
    api_sites = config.get("api_site") or {}
    if not api_sites:
        raise RuntimeError("config.json 中未配置 api_site")
    session = build_session()

    # 若指定站点，则仅在该站点搜索
    if site_key_opt:
        if site_key_opt not in api_sites:
            available = ", ".join(api_sites.keys())
            raise RuntimeError(f"未找到指定站点 `{site_key_opt}`。可用站点：{available}")
        site = api_sites[site_key_opt]
        api_root = site.get("api")
        if not api_root:
            raise RuntimeError(f"站点 `{site_key_opt}` 缺少 api 字段")
        logger.info(f"使用指定站点 `{site_key_opt}` -> {api_root}")
        search_key = query or ""
        elapsed, search_results = test_site_speed_and_search(session, api_root, search_key)
        logger.info(f"站点 `{site_key_opt}` 搜索耗时 {elapsed:.2f}s，结果数：{len(search_results)}")
        if not search_results:
            raise RuntimeError("在指定站点未搜索到结果")
        # 交互选择影片（单站点）
        first, _, _ = choose_from_results(logger, search_results, with_site=False)
        vid = str(first.get("vod_id") or first.get("id"))
        if not vid or vid == "None":
            raise RuntimeError("搜索结果未包含可用的 id")
        logger.info(f"拉取详情：ids={vid}")
        detail = fetch_detail_by_ids(session, api_root, [vid])
        detail_list = detail.get("list") or detail.get("data") or []
        if not detail_list:
            raise RuntimeError("详情数据为空")
        vod = detail_list[0]
    else:
        search_key = query or ""
        # 全站搜索聚合
        aggregated = search_all_sites(session, logger, api_sites, search_key)
        if not aggregated:
            raise RuntimeError("所有站点均无搜索结果或不可用")
        # 交互选择（包含站点来源）
        first, chosen_site_key, chosen_site_obj = choose_from_results(logger, aggregated, with_site=True)
        assert chosen_site_key is not None and chosen_site_obj is not None
        api_root = chosen_site_obj.get("api")
        if not api_root:
            raise RuntimeError(f"站点 `{chosen_site_key}` 缺少 api 字段")
        logger.info(f"选择站点：{chosen_site_key} -> {api_root}")
        vid = str(first.get("vod_id") or first.get("id"))
        if not vid or vid == "None":
            raise RuntimeError("搜索结果未包含可用的 id")
        logger.info(f"拉取详情：ids={vid}")
        detail = fetch_detail_by_ids(session, api_root, [vid])
        detail_list = detail.get("list") or detail.get("data") or []
        if not detail_list:
            raise RuntimeError("详情数据为空")
        vod = detail_list[0]
    # 以下共用下载逻辑

    # 分组/分集选择
    groups = parse_play_groups(vod)
    if groups:
        gidx, episodes = choose_episode(logger, groups)
        if len(episodes) > 1:
            logger.info(f"批量下载整组（共 {len(episodes)} 集）")
        last_output_path = None
        for idx, (epi_title, chosen_url) in enumerate(episodes, 1):
            logger.info(f"开始下载：{epi_title}")
            m3u8_url = chosen_url
            # 输出文件名
            if not output:
                safe_title = epi_title.strip().replace("/", "_")
                base_name = (vod.get("vod_name") or vod.get("name") or (query or "video")).strip().replace("/", "_")
                downloads_dir = os.path.join(os.getcwd(), "downloads")
                os.makedirs(downloads_dir, exist_ok=True)
                out_path = os.path.join(downloads_dir, f"{base_name}-{safe_title}.mp4")
            else:
                root, ext = os.path.splitext(output)
                suffix = f"-{idx:02d}"
                out_path = f"{root}{suffix}{ext or '.mp4'}"

            # 构建可用的 Referer：若聚合选择带站点对象则使用其 detail
            try:
                chosen_site_detail = locals().get("chosen_site_obj", None)
                referer_site = chosen_site_detail if isinstance(chosen_site_detail, dict) else locals().get("site", {})
                referer = (referer_site.get("detail") if isinstance(referer_site, dict) else None) or (m3u8_url.split("/", 3)[:3] and "/".join(m3u8_url.split("/", 3)[:3]))
            except Exception:
                referer = (m3u8_url.split("/", 3)[:3] and "/".join(m3u8_url.split("/", 3)[:3]))
            per_req_headers = {"Referer": referer or "", "User-Agent": DEFAULT_UA}
            session.headers.update(per_req_headers)

            # 优先 ffmpeg
            if m3u8_url.lower().endswith(".m3u8") and has_ffmpeg():
                code = download_via_ffmpeg(m3u8_url, out_path, headers=per_req_headers)
                if code != 0:
                    logger.warn("ffmpeg 下载失败，切换为内置下载器…")
                    simple_m3u8_download(session, m3u8_url, out_path)
            else:
                simple_m3u8_download(session, m3u8_url, out_path)
            logger.info(f"已完成下载：{out_path}")
            last_output_path = out_path
        return
    else:
        play_urls = parse_play_urls(vod)
        logger.info(f"解析到播放地址 {len(play_urls)} 条")
        if not play_urls:
            raise RuntimeError("未解析到播放地址")
        m3u8_url = pick_first_m3u8(play_urls) or play_urls[0]
    logger.info(f"选用播放源：{m3u8_url}")
    if not output:
        safe_name = (vod.get("vod_name") or vod.get("name") or (query or "video")).strip().replace("/", "_")
        downloads_dir = os.path.join(os.getcwd(), "downloads")
        os.makedirs(downloads_dir, exist_ok=True)
        output = os.path.join(downloads_dir, f"{safe_name}.mp4")
    else:
        parent = os.path.dirname(output)
        if parent:
            os.makedirs(parent, exist_ok=True)

    # 构建可用的 Referer：若聚合选择带站点对象则使用其 detail
    # 由于上方逻辑可能来自两种分支，这里尽量回退使用 m3u8 域名
    try:
        chosen_site_detail = locals().get("chosen_site_obj", None)
        referer_site = chosen_site_detail if isinstance(chosen_site_detail, dict) else locals().get("site", {})
        referer = (referer_site.get("detail") if isinstance(referer_site, dict) else None) or (m3u8_url.split("/", 3)[:3] and "/".join(m3u8_url.split("/", 3)[:3]))
    except Exception:
        referer = (m3u8_url.split("/", 3)[:3] and "/".join(m3u8_url.split("/", 3)[:3]))
    # 为会话设置专用 headers（对 m3u8/分片）
    per_req_headers = {"Referer": referer or "", "User-Agent": DEFAULT_UA}
    session.headers.update(per_req_headers)

    logger.info("开始下载…")
    # 优先 ffmpeg
    if m3u8_url.lower().endswith(".m3u8") and has_ffmpeg():
        code = download_via_ffmpeg(m3u8_url, output, headers=per_req_headers)
        if code != 0:
            # 失败则尝试内置简易下载
            logger.warn("ffmpeg 下载失败，切换为内置下载器…")
            simple_m3u8_download(session, m3u8_url, output)
    else:
        simple_m3u8_download(session, m3u8_url, output)

    logger.info(f"已完成下载：{output}")


def main():
    parser = argparse.ArgumentParser(description="MoonTV 风格数据源的视频搜索与下载工具")
    parser.add_argument("-q", "--query", required=True, help="搜索关键字，支持模糊搜索")
    parser.add_argument("-o", "--output", help="输出文件名，默认使用影片名.mp4")
    parser.add_argument("-c", "--config", default="config.json", help="配置文件路径，默认当前目录 config.json")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出详细执行状态")
    parser.add_argument("-s", "--site", help="指定站点 key（来自 config.json 的 api_site）")
    args = parser.parse_args()

    try:
        run(args.output, args.config, verbose=args.verbose, site_key_opt=args.site, query=args.query)
    except Exception as e:
        print(f"错误：{e}")
        sys.exit(1)


if __name__ == "__main__":
    main()


