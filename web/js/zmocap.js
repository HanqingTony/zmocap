/**
 * zmocap 前端扩展：
 * - “VRM Loader”节点内置 3D 预览：上传或下拉选择后立刻加载显示，不排队、不跑工作流；
 * - “Rig”节点内置 3D 预览：跑完自动播放，`skinned` 开关是纯前端图层切换，即时生效。
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { createViewer, inputFileUrl, outputFileUrl } from "./zmocap_viewer.js";

const CHARACTER_NODES = new Set(["ZMocapVRMLoader"]);
const RIG_NODES = new Set(["ZMocapRig", "ZMocapSMPLPreviewer", "ZMocapVRMPreviewer"]);

function addPreviewWidget(node, viewer, container, title) {
  const hint = document.createElement("div");
  hint.textContent = title;
  hint.style.cssText =
    "position:absolute;inset:auto 0 6px 0;text-align:center;font-size:11px;color:#8fa3b8;" +
    "pointer-events:none;user-select:none";
  container.appendChild(hint);
  const widget = node.addDOMWidget("zmocap_preview", "zmocap3d", container, {
    serialize: false,
    hideOnZoom: false,
    getMinHeight: () => 260,
  });
  widget.computeSize = (width) => [width, 268];
  void viewer;
  node.setSize([Math.max(node.size[0], 320), node.size[1] + 268]);
  return { container, hint, widget };
}

app.registerExtension({
  name: "zmocap.viewer",

  async beforeRegisterNodeDef(nodeType, nodeData) {
    const isCharacter = CHARACTER_NODES.has(nodeData.name);
    const isRig = RIG_NODES.has(nodeData.name);
    if (!isCharacter && !isRig) {
      return;
    }

    const onCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      onCreated?.apply(this, arguments);
      const node = this;
      const container = document.createElement("div");
      container.style.cssText =
        "width:100%;height:260px;min-height:160px;border-radius:6px;overflow:hidden;position:relative;" +
        "background:radial-gradient(circle at 50% 30%, #2a3340, #14181f 70%);";
      const viewer = createViewer(container);
      node.__zmocapViewer = viewer;
      const { hint } = addPreviewWidget(node, viewer, container, isCharacter
        ? "选择或上传 VRM 即预览"
        : "运行后在此播放动作（skinned 开关即时切换显示层）");

      if (isCharacter) {
        const combo = node.widgets?.find((w) => w.name === "character");
        if (combo) {
          const original = combo.callback;
          combo.callback = function (...args) {
            const result = original?.apply(this, args);
            loadCharacter(node, combo.value, hint);
            return result;
          };
          setTimeout(() => loadCharacter(node, combo.value, hint), 60);
        }
        addUploadButton(node, hint);
      }

      if (isRig) {
        const skinned = node.widgets?.find((w) => w.name === "skinned");
        const apply = () => viewer.setLayer(skinned?.value === false ? "bones" : "skinned");
        if (skinned) {
          const original = skinned.callback;
          skinned.callback = function (...args) {
            const result = original?.apply(this, args);
            apply();
            return result;
          };
        }
        apply();
        addPlayPauseButton(node, viewer);
      }

      const onRemoved = nodeType.prototype.onRemoved;
      nodeType.prototype.onRemoved = function () {
        viewer.dispose();
        onRemoved?.apply(this, arguments);
      };
    };

    if (isRig) {
      const onExecuted = nodeType.prototype.onExecuted;
      nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);
        const name = message?.zmocap_glb?.[0];
        if (!name) return;
        const viewer = this.__zmocapViewer;
        const skinned = this.widgets?.find((w) => w.name === "skinned");
        viewer?.load(outputFileUrl(name)).then((info) => {
          viewer.setLayer(skinned?.value === false ? "bones" : "skinned");
          const hint = this.widgets?.find((w) => w.name === "zmocap_preview");
          void hint;
          console.log(`[zmocap] 预览已载入 ${name}（动画 ${info?.animations || 0} 条）`);
        });
      };
    }
  },
});

function loadCharacter(node, value, hint) {
  const viewer = node.__zmocapViewer;
  if (!viewer) return;
  if (!value || value === "none") {
    viewer.reset();
    if (hint) hint.textContent = "未选择 VRM（输出 none）";
    return;
  }
  if (hint) hint.textContent = "载入中…";
  viewer.load(inputFileUrl(value)).then((info) => {
    if (hint) {
      hint.textContent = info ? `已载入 ${value.split("/").pop()}` : `载入失败：${value}`;
    }
  });
}

function addUploadButton(node, hint) {
  const button = node.addWidget("button", "上传 VRM 文件", null, async () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".vrm,.glb,.gltf";
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) return;
      const body = new FormData();
      body.append("image", file, file.name);
      body.append("type", "input");
      body.append("subfolder", "zmocap");
      body.append("overwrite", "true");
      const response = await api.fetchApi("/upload/image", { method: "POST", body });
      if (!response.ok) {
        alert(`上传失败：${response.status} ${await response.text()}`);
        return;
      }
      const data = await response.json();
      const relative = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
      const combo = node.widgets?.find((w) => w.name === "character");
      if (combo) {
        const options = combo.options?.values;
        const list = Array.isArray(options) ? options : [...(options || [])];
        if (!list.includes(relative)) {
          list.push(relative);
          if (combo.options) combo.options.values = list;
        }
        combo.value = relative;
      }
      node.setDirtyCanvas?.(true, true);
      loadCharacter(node, relative, hint);
    };
    input.click();
  });
  button.serialize = false;
}

function addPlayPauseButton(node, viewer) {
  let playing = true;
  const button = node.addWidget("button", "播放 / 暂停", null, () => {
    playing = !playing;
    viewer.setPlaying(playing);
    button.name = playing ? "暂停" : "播放";
    node.setDirtyCanvas?.(true, true);
  });
  button.serialize = false;
}
