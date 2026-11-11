"""生成 ComfyUI 工作流模板（UI 格式），输出到桌面。

用法::

    python scripts/make_workflow.py ~/Desktop/zmocap-exmr.json [视频文件名]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BONES_UI = None


def build(video: str = "zmocap_test.mp4") -> dict:
    nodes: list[dict] = []
    links: list[list] = []
    counter = {"link": 0, "node": 0}

    def add(kind, pos, size, widgets, outputs=None, inputs=None, title=None):
        node_id = counter["node"]
        counter["node"] += 1
        node = {
            "id": node_id,
            "type": kind,
            "pos": list(pos),
            "size": list(size),
            "flags": {},
            "order": node_id,
            "mode": 0,
            "inputs": inputs or [],
            "outputs": outputs or [],
            "properties": {"Node name for S&R": kind},
        }
        if widgets:
            node["widgets_values"] = widgets
        if title:
            node["title"] = title
        nodes.append(node)
        return node_id

    def link(src, src_slot, dst, dst_slot, type_):
        counter["link"] += 1
        links.append([counter["link"], src, src_slot, dst, dst_slot, type_])
        return counter["link"]

    # 1) 官方视频节点
    load = add("LoadVideo", (40, 60), (300, 90), [video],
               outputs=[{"name": "VIDEO", "type": "VIDEO", "links": []}])
    get = add("GetVideoComponents", (400, 60), (300, 120), [],
              outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []},
                       {"name": "AUDIO", "type": "AUDIO", "links": []},
                       {"name": "FPS", "type": "FLOAT", "links": []}],
              inputs=[{"name": "video", "type": "VIDEO", "link": None}])
    link(load, 0, get, 0, "VIDEO")
    nodes[get]["inputs"][0]["link"] = links[-1][0]
    nodes[load]["outputs"][0]["links"].append(links[-1][0])

    # 2) zmocap：跟踪 / 帧落盘 / 模型
    track = add("ZMocapTrack", (760, 60), (300, 170), ["yolov8x.pt", 0.3, 55.0, 0],
                outputs=[{"name": "tracks", "type": "ZMOCAP_TRACKS", "links": []}],
                inputs=[{"name": "images", "type": "IMAGE", "link": None}])
    frames = add("ZMocapFrames", (760, 300), (300, 100), [95],
                 outputs=[{"name": "frames", "type": "ZMOCAP_FRAMES", "links": []}],
                 inputs=[{"name": "images", "type": "IMAGE", "link": None}])
    models = add("ZMocapModels", (760, 450), (300, 100), ["", True],
                 outputs=[{"name": "models", "type": "ZMOCAP_MODELS", "links": []}])

    for dst, slot in ((track, 0), (frames, 0)):
        link(get, 0, dst, slot, "IMAGE")
        nodes[dst]["inputs"][slot]["link"] = links[-1][0]
        nodes[get]["outputs"][0]["links"].append(links[-1][0])

    # 3) 三个特征阶段
    keypoints = add("ZMocapSapiens2Keypoints", (1120, 60), (320, 120), [True],
                    outputs=[{"name": "keypoints", "type": "ZMOCAP_KEYPOINTS", "links": []}],
                    inputs=[{"name": "frames", "type": "ZMOCAP_FRAMES", "link": None},
                            {"name": "tracks", "type": "ZMOCAP_TRACKS", "link": None},
                            {"name": "models", "type": "ZMOCAP_MODELS", "link": None}])
    body = add("ZMocapSam3DBodyFeatures", (1120, 240), (320, 120), [True],
               outputs=[{"name": "body_features", "type": "ZMOCAP_BODY_FEATURES", "links": []}],
               inputs=[{"name": "frames", "type": "ZMOCAP_FRAMES", "link": None},
                       {"name": "tracks", "type": "ZMOCAP_TRACKS", "link": None},
                       {"name": "models", "type": "ZMOCAP_MODELS", "link": None}])
    hands = add("ZMocapHaMeRHandFeatures", (1120, 420), (320, 140), [True],
                outputs=[{"name": "hand_features", "type": "ZMOCAP_HAND_FEATURES", "links": []}],
                inputs=[{"name": "frames", "type": "ZMOCAP_FRAMES", "link": None},
                        {"name": "tracks", "type": "ZMOCAP_TRACKS", "link": None},
                        {"name": "models", "type": "ZMOCAP_MODELS", "link": None},
                        {"name": "keypoints", "type": "ZMOCAP_KEYPOINTS", "link": None}])

    def wire(src, src_slot, dst, dst_slot, type_):
        link(src, src_slot, dst, dst_slot, type_)
        nodes[dst]["inputs"][dst_slot]["link"] = links[-1][0]
        nodes[src]["outputs"][src_slot]["links"].append(links[-1][0])

    for target in (keypoints, body, hands):
        wire(frames, 0, target, 0, "ZMOCAP_FRAMES")
        wire(track, 0, target, 1, "ZMOCAP_TRACKS")
        wire(models, 0, target, 2, "ZMOCAP_MODELS")
    wire(keypoints, 0, hands, 3, "ZMOCAP_KEYPOINTS")

    # 4) 解算
    exmr = add("ZMocapExMR", (1500, 60), (320, 220), [24.0, True, True],
               outputs=[{"name": "motion", "type": "ZMOCAP_MOTION", "links": []}],
               inputs=[{"name": "keypoints", "type": "ZMOCAP_KEYPOINTS", "link": None},
                       {"name": "body_features", "type": "ZMOCAP_BODY_FEATURES", "link": None},
                       {"name": "hand_features", "type": "ZMOCAP_HAND_FEATURES", "link": None},
                       {"name": "tracks", "type": "ZMOCAP_TRACKS", "link": None},
                       {"name": "models", "type": "ZMOCAP_MODELS", "link": None},
                       {"name": "fps", "type": "FLOAT", "link": None}])
    wire(keypoints, 0, exmr, 0, "ZMOCAP_KEYPOINTS")
    wire(body, 0, exmr, 1, "ZMOCAP_BODY_FEATURES")
    wire(hands, 0, exmr, 2, "ZMOCAP_HAND_FEATURES")
    wire(track, 0, exmr, 3, "ZMOCAP_TRACKS")
    wire(models, 0, exmr, 4, "ZMOCAP_MODELS")
    wire(get, 2, exmr, 5, "FLOAT")

    # 5) 绑定 + 预览 + 导出
    rig = add("ZMocapRig", (1880, 60), (320, 140), ["zmocap_rig", "", "auto"],
              outputs=[{"name": "glb_path", "type": "STRING", "links": []}],
              inputs=[{"name": "motion", "type": "ZMOCAP_MOTION", "link": None}])
    wire(exmr, 0, rig, 0, "ZMOCAP_MOTION")

    preview3d = add("Preview3D", (1880, 260), (380, 420), ["zmocap_rig.glb"])
    skeleton = add("ZMocapPreview3D", (1880, 720), (360, 420), [600],
                   inputs=[{"name": "motion", "type": "ZMOCAP_MOTION", "link": None}])
    wire(exmr, 0, skeleton, 0, "ZMOCAP_MOTION")

    export = add("ZMocapExport", (2240, 60), (300, 140), ["zmocap_motion", True],
                 outputs=[{"name": "saved_path", "type": "STRING", "links": []}],
                 inputs=[{"name": "motion", "type": "ZMOCAP_MOTION", "link": None}])
    wire(exmr, 0, export, 0, "ZMOCAP_MOTION")

    return {
        "last_node_id": counter["node"] - 1,
        "last_link_id": counter["link"],
        "nodes": nodes,
        "links": links,
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4,
    }


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "zmocap-exmr.json").expanduser()
    video = sys.argv[2] if len(sys.argv) > 2 else "zmocap_test.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(video), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}")
