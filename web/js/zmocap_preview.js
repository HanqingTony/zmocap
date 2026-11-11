import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// SMPL-X 55 关节父索引（含 15 指骨/手），与后端 SMPLX_PARENTS 一致
const PARENTS = [
  -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
  15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 20, 37, 38,
  21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52, 53,
];
const BONES = [];
for (let i = 1; i < PARENTS.length; i++) BONES.push([PARENTS[i], i]);

// 手指单独加粗显示的颜色分组
const FINGER = new Set();
for (let i = 25; i < 55; i++) FINGER.add(i);

function draw(canvas, data) {
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.fillStyle = "#111";
  ctx.fillRect(0, 0, w, h);
  if (!data || !data.joints || !data.joints.length) {
    ctx.fillStyle = "#888";
    ctx.font = "13px sans-serif";
    ctx.fillText("运行后显示骨骼", 10, 20);
    return;
  }

  const joints = data.joints;
  const frames = joints.length;
  const jointCount = Math.min(joints[0].length, 55);
  for (let f = 0; f < frames; f++) {
    for (let j = 0; j < jointCount; j++) {
      const p = joints[f][j];
      if (!p) continue;
      const minX = Math.min(...joints.map((fr) => Math.min(...fr.slice(0, jointCount).map((q) => q[0]))));
      const maxX = Math.max(...joints.map((fr) => Math.max(...fr.slice(0, jointCount).map((q) => q[0]))));
      const minY = Math.min(...joints.map((fr) => Math.min(...fr.slice(0, jointCount).map((q) => q[1]))));
      const maxY = Math.max(...joints.map((fr) => Math.max(...fr.slice(0, jointCount).map((q) => q[1]))));
      const pad = 14;
      const sx = (x) => pad + ((x - minX) / Math.max(maxX - minX, 1e-6)) * (w - pad * 2);
      const sy = (y) => pad + ((y - minY) / Math.max(maxY - minY, 1e-6)) * (h - pad * 2);

      ctx.lineWidth = f === data.frame ? 2.2 : 1;
      ctx.strokeStyle = f === data.frame ? "#39f" : "rgba(120,170,255,0.35)";
      ctx.beginPath();
      for (const [a, b] of BONES) {
        if (a >= jointCount || b >= jointCount) continue;
        ctx.moveTo(sx(joints[f][a][0]), sy(joints[f][a][1]));
        ctx.lineTo(sx(joints[f][b][0]), sy(joints[f][b][1]));
      }
      ctx.stroke();

      if (f === data.frame) {
        for (let k = 0; k < jointCount; k++) {
          const q = joints[f][k];
          ctx.fillStyle = FINGER.has(k) ? "#ffd23f" : "#7ee787";
          const r = k === 0 ? 5 : FINGER.has(k) ? 2.6 : 3.4;
          ctx.beginPath();
          ctx.arc(sx(q[0]), sy(q[1]), r, 0, Math.PI * 2);
          ctx.fill();
        }
        ctx.fillStyle = "#eee";
        ctx.font = "12px monospace";
        const text = `frame ${f + 1}/${frames} · ${data.fps?.toFixed(1) ?? "?"} fps`;
        ctx.fillText(text, 10, h - 8);
      }
    }
  }
}

app.registerExtension({
  name: "zmocap.preview",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "ZMocapPreview3D") return;

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      onExecuted?.apply(this, arguments);
      const payload = message?.zmocap_joints?.[0];
      if (!payload) return;
      this.__zmocap = {
        joints: payload.joints,
        fps: payload.fps,
        frame: 0,
        timer: null,
      };
      if (!this.__zmocapCanvas) {
        const canvas = document.createElement("canvas");
        canvas.width = 320;
        canvas.height = 320;
        canvas.style.cssText = "width:100%;border-radius:6px;display:block;margin-top:6px";
        this.addDOMWidget("zmocap_preview", "canvas", canvas, {
          serialize: false,
          hideOnZoom: false,
        });
        this.__zmocapCanvas = canvas;
      }
      draw(this.__zmocapCanvas, { ...this.__zmocap, frame: 0 });
      clearInterval(this.__zmocap.timer);
      this.__zmocap.timer = setInterval(() => {
        const state = this.__zmocap;
        if (!state) return;
        state.frame = (state.frame + 1) % state.joints.length;
        draw(this.__zmocapCanvas, state);
      }, Math.max(24, 1000 / (this.__zmocap.fps || 30)));
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      if (this.__zmocap?.timer) clearInterval(this.__zmocap.timer);
      onRemoved?.apply(this, arguments);
    };
  },
});
