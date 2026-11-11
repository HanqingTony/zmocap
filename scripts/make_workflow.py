"""生成 ComfyUI 工作流模板（UI 格式），默认写到桌面。

结构（motion 之后两条分支：一条看、一条存）::

    LoadVideo → GetVideoComponents → ZMocapTrack / ZMocapFrames / ZMocapModels
      → ZMocapSapiens2Keypoints / ZMocapSam3DBodyFeatures / ZMocapHaMeRHandFeatures
      → ZMocapExMR → ZMocapRig ← ZMocapVRMLoader（两者都在节点内预览）
                    └→ ZMocapExport

角色加载（ZMocapVRMLoader）与动作绑定（ZMocapRig）完全解耦。

用法::

    python scripts/make_workflow.py <输出路径> [视频文件名] [角色文件名]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


class Builder:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.links: list[list] = []

    def node(self, kind, pos, size, widgets=None, inputs=(), outputs=(), title=None):
        node_id = len(self.nodes)
        self.nodes.append({
            "id": node_id, "type": kind, "pos": list(pos), "size": list(size),
            "flags": {}, "order": node_id, "mode": 0,
            "inputs": [{"name": n, "type": t, "link": None} for n, t in inputs],
            "outputs": [{"name": n, "type": t, "links": []} for n, t in outputs],
            "properties": {"Node name for S&R": kind},
            "widgets_values": widgets if widgets is not None else [],
            **({"title": title} if title else {}),
        })
        return node_id

    def link(self, src, slot, dst, port, type_):
        self.links.append([len(self.links) + 1, src, slot, dst, port, type_])
        self.nodes[dst]["inputs"][port]["link"] = self.links[-1][0]
        self.nodes[src]["outputs"][slot]["links"].append(self.links[-1][0])

    def dump(self) -> dict:
        return {"last_node_id": len(self.nodes) - 1, "last_link_id": len(self.links),
                "nodes": self.nodes, "links": self.links, "groups": [], "config": {},
                "extra": {}, "version": 0.4}


def build(video: str = "zmocap_test.mp4", character: str = "(none / built-in skeleton)") -> dict:
    b = Builder()

    load = b.node("LoadVideo", (40, 80), (320, 100), [video],
                  outputs=[("VIDEO", "VIDEO")])
    parts = b.node("GetVideoComponents", (400, 80), (320, 130), [],
                   inputs=[("video", "VIDEO")],
                   outputs=[("IMAGE", "IMAGE"), ("AUDIO", "AUDIO"), ("FPS", "FLOAT"),
                            ("WIDTH", "INT"), ("HEIGHT", "INT")])
    b.link(load, 0, parts, 0, "VIDEO")

    track = b.node("ZMocapTrack", (760, 80), (310, 180), ["yolov8x.pt", 0.3, 55.0, 0],
                   inputs=[("images", "IMAGE")], outputs=[("tracks", "ZMOCAP_TRACKS")])
    frames = b.node("ZMocapFrames", (760, 300), (310, 100), [95],
                    inputs=[("images", "IMAGE")], outputs=[("frames", "ZMOCAP_FRAMES")])
    models = b.node("ZMocapModels", (760, 440), (310, 100), ["", True],
                    outputs=[("models", "ZMOCAP_MODELS")])
    b.link(parts, 0, track, 0, "IMAGE")
    b.link(parts, 0, frames, 0, "IMAGE")

    def stage(kind, pos, extra_inputs=(), extra_widgets=None, outputs=(("out", "OUT"),)):
        node = b.node(kind, pos, (330, 140), extra_widgets if extra_widgets is not None else [True],
                      inputs=[("frames", "ZMOCAP_FRAMES"), ("tracks", "ZMOCAP_TRACKS"),
                              ("models", "ZMOCAP_MODELS"), *extra_inputs],
                      outputs=list(outputs))
        b.link(frames, 0, node, 0, "ZMOCAP_FRAMES")
        b.link(track, 0, node, 1, "ZMOCAP_TRACKS")
        b.link(models, 0, node, 2, "ZMOCAP_MODELS")
        return node

    keypoints = stage("ZMocapSapiens2Keypoints", (1130, 80),
                      extra_widgets=[True, "448x336"],
                      outputs=[("keypoints", "ZMOCAP_KEYPOINTS")])
    body = stage("ZMocapSam3DBodyFeatures", (1130, 260),
                 outputs=[("body_features", "ZMOCAP_BODY_FEATURES")])
    hands = stage("ZMocapHaMeRHandFeatures", (1130, 440),
                  extra_inputs=[("keypoints", "ZMOCAP_KEYPOINTS")],
                  outputs=[("hand_features", "ZMOCAP_HAND_FEATURES")])
    b.link(keypoints, 0, hands, 3, "ZMOCAP_KEYPOINTS")

    exmr = b.node("ZMocapExMR", (1510, 80), (330, 260), [30.0, True, True, False, False],
                  inputs=[("keypoints", "ZMOCAP_KEYPOINTS"), ("body_features", "ZMOCAP_BODY_FEATURES"),
                          ("hand_features", "ZMOCAP_HAND_FEATURES"), ("tracks", "ZMOCAP_TRACKS"),
                          ("models", "ZMOCAP_MODELS"), ("fps", "FLOAT")],
                  outputs=[("motion", "ZMOCAP_MOTION")])
    b.link(keypoints, 0, exmr, 0, "ZMOCAP_KEYPOINTS")
    b.link(body, 0, exmr, 1, "ZMOCAP_BODY_FEATURES")
    b.link(hands, 0, exmr, 2, "ZMOCAP_HAND_FEATURES")
    b.link(track, 0, exmr, 3, "ZMOCAP_TRACKS")
    b.link(models, 0, exmr, 4, "ZMOCAP_MODELS")
    b.link(parts, 2, exmr, 5, "FLOAT")

    # 角色加载独立成节点：只负责“选/上传角色 → 角色对象”，不看动作。
    loader = b.node("ZMocapVRMLoader", (1900, 380), (330, 90), [character],
                    outputs=[("character", "ZMOCAP_CHARACTER"), ("glb_path", "STRING")])

    # 绑定：motion + 角色 → GLB（唯一消费者关系：Rig 吃 motion，吃角色）
    rig = b.node("ZMocapRig", (2280, 80), (330, 120), ["zmocap_rig", True],
                 inputs=[("motion", "ZMOCAP_MOTION"), ("character", "ZMOCAP_CHARACTER")],
                 outputs=[("glb_path", "STRING")])
    b.link(exmr, 0, rig, 0, "ZMOCAP_MOTION")
    b.link(loader, 0, rig, 1, "ZMOCAP_CHARACTER")


    # SOMA 枢纽：SMPL-X → SOMA(78 关节) → 角色（可换成其它目标的同类节点）
    to_soma = b.node("ZMocapSMPL2SOMA", (2280, 500), (330, 120), [30.0, True],
                     inputs=[("params", "ZMOCAP_MOTION")],
                     outputs=[("soma", "ZMOCAP_SOMA"), ("info", "STRING")])
    b.link(exmr, 0, to_soma, 0, "ZMOCAP_MOTION")

    soma_rig = b.node("ZMocapSOMA2VRM", (2640, 500), (330, 150), [30.0, True],
                      inputs=[("soma", "ZMOCAP_SOMA"), ("character", "ZMOCAP_CHARACTER")],
                      outputs=[("glb", "STRING"), ("info", "STRING")])
    b.link(to_soma, 0, soma_rig, 0, "ZMOCAP_SOMA")
    b.link(loader, 0, soma_rig, 1, "ZMOCAP_CHARACTER")

    export = b.node("ZMocapExport", (2280, 720), (330, 130), ["zmocap_motion", True],
                    inputs=[("motion", "ZMOCAP_MOTION")], outputs=[("saved_path", "STRING")])
    b.link(exmr, 0, export, 0, "ZMOCAP_MOTION")

    return b.dump()


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "zmocap-exmr.json").expanduser()
    video = sys.argv[2] if len(sys.argv) > 2 else "zmocap_test.mp4"
    character = sys.argv[3] if len(sys.argv) > 3 else "(none / built-in skeleton)"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build(video, character), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写入 {out}")
