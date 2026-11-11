// zmocap 3D 预览：把节点输出的关节序列画在节点内部的 canvas 上（无外部依赖）。
import { app } from "../../scripts/app.js";

const BONES = [
  [0, 1], [1, 2], [2, 5], [5, 8], [8, 11],           // 脊柱
  [1, 4], [4, 7], [7, 10],                            // 右臂
  [1, 3], [3, 6], [6, 9], [9, 12], [12, 15],          // 左臂/手
  [9, 13], [13, 16], [16, 18], [18, 20],              // 左手
  [9, 14], [14, 17], [17, 19], [19, 21],              // 右手
  [0, 1], [2, 5],
  [0, 2], [2, 5],
  [5, 8], [8, 11],
  [11, 12], [12, 13], [13, 14], [14, 15],             // 下颌/头
  [11, 22], [22, 23], [23, 24],                       // 右脚链
  [11, 25], [25, 26], [26, 27],                       // 左脚链
];

app.registerExtension({
  name: "zmocap.preview3d",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== "ZMocapPreview3D") return;

    const original = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      original?.apply(this, arguments);
      const joints = message?.zmocap_joints?.[0];
      if (!joints || !joints.length) return;
      this._zmocap = {
        joints,
        fps: message.zmocap_fps?.[0] ?? 30,
        frame: 0,
        yaw: 0.35,
        pitch: -0.15,
        zoom: 1.0,
        playing: true,
      };
      if (!this._zmocapTimer) {
        this._zmocapTimer = setInterval(() => {
          if (!this._zmocap) return;
          if (this._zmocap.playing) this._zmocap.frame = (this._zmocap.frame + 1) % this._zmocap.joints.length;
          draw(this);
        }, Math.max(16, 1000 / this._zmocap.fps));
      }
      this.setDirtyCanvas?.(true, true);
    };

    const onDrawBackground = nodeType.prototype.onDrawBackground;
    nodeType.prototype.onDrawBackground = function (ctx) {
      onDrawBackground?.apply(this, arguments);
      if (this.flags?.collapsed || !this._zmocap) return;
      const [w, h] = this.size;
      const box = [12, 26, w - 24, h - 38];
      ctx.save();
      ctx.fillStyle = "#101418";
      ctx.fillRect(...box);
      ctx.strokeStyle = "#2a3138";
      ctx.strokeRect(...box);
      drawSkeleton(ctx, box, this._zmocap);
      ctx.restore();
    };

    const onMouseDown = nodeType.prototype.onMouseDown;
    nodeType.prototype.onMouseDown = function (event, pos, canvas) {
      const [w, h] = this.size;
      if (pos[0] > 12 && pos[0] < w - 12 && pos[1] > 26 && pos[1] < h - 12) {
        this._zmocapDrag = pos;
        if (this._zmocap) this._zmocap.playing = !this._zmocap.playing;
        return true;
      }
      return onMouseDown?.apply(this, arguments);
    };

    const onMouseMove = nodeType.prototype.onMouseMove;
    nodeType.prototype.onMouseMove = function (event, pos, canvas) {
      if (this._zmocapDrag && this._zmocap) {
        this._zmocap.yaw += (pos[0] - this._zmocapDrag[0]) * 0.01;
        this._zmocap.pitch = Math.max(-1.2, Math.min(1.2, this._zmocap.pitch + (pos[1] - this._zmocapDrag[1]) * 0.01));
        this._zmocapDrag = pos;
        return true;
      }
      return onMouseMove?.apply(this, arguments);
    };

    const onMouseUp = nodeType.prototype.onMouseUp;
    nodeType.prototype.onMouseUp = function () {
      this._zmocapDrag = null;
      return onMouseUp?.apply(this, arguments);
    };
  },
});

function draw(node) {
  node.setDirtyCanvas?.(true, false);
}

function drawSkeleton(ctx, [x, y, w, h], state) {
  const joints = state.joints[state.frame % state.joints.length];
  if (!joints) return;
  const count = joints.length;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (let i = 0; i < 24 && i < count; i++) {
    minX = Math.min(minX, joints[i][0]); maxX = Math.max(maxX, joints[i][0]);
    minY = Math.min(minY, joints[i][1]); maxY = Math.max(maxY, joints[i][1]);
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const scale = (Math.min(w, h) / Math.max(maxX - minX, maxY - minY, 0.5)) * 0.42 * state.zoom;
  const cosY = Math.cos(state.yaw), sinY = Math.sin(state.yaw);
  const cosP = Math.cos(state.pitch), sinP = Math.sin(state.pitch);

  const project = (p) => {
    const px = (p[0] - cx), py = (p[1] - cy), pz = p[2];
    const rx = px * cosY + pz * sinY;
    const rz = -px * sinY + pz * cosY;
    const ry = py * cosP - rz * sinP;
    const depth = py * sinP + rz * cosP;
    return [x + w / 2 + rx * scale, y + h / 2 - ry * scale, depth];
  };

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#5fd0c8";
  ctx.beginPath();
  for (const [a, b] of BONES) {
    if (a >= count || b >= count) continue;
    const pa = project(joints[a]);
    const pb = project(joints[b]);
    ctx.moveTo(pa[0], pa[1]);
    ctx.lineTo(pb[0], pb[1]);
  }
  ctx.stroke();

  ctx.fillStyle = "#f2f5f7";
  for (let i = 0; i < count && i < 24; i++) {
    const p = project(joints[i]);
    ctx.beginPath();
    ctx.arc(p[0], p[1], 2.5, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.fillStyle = "#8b98a5";
  ctx.font = "11px monospace";
  ctx.fillText(`frame ${state.frame + 1}/${state.joints.length}  fps ${state.fps.toFixed(0)}`, x + 8, y + h - 8);
}
