/**
 * zmocap 内置 3D 预览器：给“VRM Loader”和“Rig”节点共用的 three.js 视图。
 *
 * 设计要点：
 * - 纯前端渲染，切换显示层（蒙皮网格 / 骨骼）不需要重跑工作流；
 * - 没有 setInterval 轮询，只在 requestAnimationFrame 里按需渲染；
 * - 每帧最多一次 rAF，节点不可见（宽度为 0）时自动跳过，避免后端执行时前端抢 CPU。
 */

import * as THREE from "./vendor/three.module.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const loader = new GLTFLoader();

/** 在容器里挂一个预览器，返回操作句柄。 */
export function createViewer(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  container.appendChild(renderer.domElement);
  renderer.domElement.style.cssText = "display:block;width:100%;height:100%";

  const scene = new THREE.Scene();
  scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 2.2));
  const key = new THREE.DirectionalLight(0xffffff, 2.0);
  key.position.set(1.5, 3.0, 2.0);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.8);
  fill.position.set(-2.0, 1.0, -1.5);
  scene.add(fill);

  const camera = new THREE.PerspectiveCamera(35, 1, 0.01, 200);
  camera.position.set(0, 1.2, 3.4);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 1.0, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.update();

  const grid = new THREE.GridHelper(6, 12, 0x445566, 0x223344);
  grid.material.opacity = 0.5;
  grid.material.transparent = true;
  scene.add(grid);

  let mixer = null;
  let current = null;
  let playing = true;
  let disposed = false;
  let frames = [];

  const clock = new THREE.Clock();

  function resize() {
    const width = container.clientWidth || 0;
    const height = container.clientHeight || 0;
    if (!width || !height) return false;
    const changed = renderer.domElement.width !== Math.floor(width * renderer.getPixelRatio()) ||
      renderer.domElement.height !== Math.floor(height * renderer.getPixelRatio());
    if (!changed) return true;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    return true;
  }

  function frame() {
    if (disposed) return;
    window.requestAnimationFrame(frame);
    if (!container.isConnected || !container.clientWidth) return; // 折叠/隐藏时不渲染
    frames.push(1);
    if (frames.length > 120) {
      frames = [];
      resize();
    }
    const delta = clock.getDelta();
    if (mixer && playing) mixer.update(delta);
    controls.update();
    renderer.render(scene, camera);
  }
  window.requestAnimationFrame(frame);

  function clear() {
    if (!current) return;
    scene.remove(current);
    current.traverse((object) => {
      object.geometry?.dispose?.();
      const material = object.material;
      if (Array.isArray(material)) material.forEach((item) => item.dispose?.());
      else material?.dispose?.();
    });
    current = null;
    mixer = null;
  }

  function look(scaleHint) {
    const box = new THREE.Box3().setFromObject(current);
    if (box.isEmpty()) return;
    const size = new THREE.Vector3();
    const center = new THREE.Vector3();
    box.getSize(size);
    box.getCenter(center);
    const height = Math.max(size.y, 0.5);
    camera.position.set(center.x, center.y + height * 0.12, center.z + height * 1.55);
    controls.target.copy(center);
    controls.update();
    grid.position.set(center.x, box.min.y, center.z);
    if (scaleHint) renderer.render(scene, camera);
  }

  return {
    /** 载入 GLB / VRM（VRM 会当作普通 glTF 解析，材质回退到标准材质）。 */
    async load(url, { autoplay = true } = {}) {
      if (!url) return null;
      try {
        const gltf = await loader.loadAsync(url);
        if (disposed) return null;
        clear();
        current = gltf.scene || gltf.scenes?.[0];
        if (!current) return null;
        current.traverse((object) => {
          if (object.isSkinnedMesh) object.frustumCulled = false;
        });
        scene.add(current);
        if (gltf.animations?.length) {
          mixer = new THREE.AnimationMixer(current);
          for (const clip of gltf.animations) mixer.clipAction(clip).play();
          playing = autoplay;
        }
        this.setLayer(this.layer || "skinned");
        look(true);
        return { animations: gltf.animations?.length || 0 };
      } catch (error) {
        console.warn("[zmocap] 载入失败", url, error);
        return null;
      }
    },

    /** 显示层切换：skinned 显示蒙皮网格，bones 只显示骨骼小棍/点线。 */
    setLayer(layer) {
      this.layer = layer;
      if (!current) return;
      current.traverse((object) => {
        const isStick = typeof object.name === "string" && object.name.startsWith("zmocap_stick");
        const boneLayer = isStick || object.userData?.zmocapBone === true ||
          object.userData?.zmocapBone === "true";
        if (boneLayer) {
          object.visible = layer === "bones";
        } else if (object.isSkinnedMesh || object.isMesh) {
          // 骨骼视图下隐藏角色蒙皮，避免"皮肤和棍子叠在一起"
          object.visible = layer === "skinned";
        } else if (object.isPoints || object.isLine) {
          object.visible = layer === "bones";
        }
      });
    },

    /** 骨骼点/线是否存在于模型里（用于决定 UI 是否提示）。 */
    hasBones() {
      if (!current) return false;
      let found = false;
      current.traverse((object) => {
        if (object.isPoints || object.isLine) found = true;
      });
      return found;
    },

    setPlaying(value) {
      playing = value;
      if (mixer) mixer.timeScale = value ? 1 : 0;
    },

    reset() {
      clear();
      renderer.render(scene, camera);
    },

    dispose() {
      disposed = true;
      clear();
      controls.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    },
  };
}

/** 把 ComfyUI 的 /view 地址按输入文件路径拼出来。 */
export function inputFileUrl(value, type = "input") {
  if (!value) return null;
  const normalized = String(value).replace(/\\/g, "/");
  const slash = normalized.lastIndexOf("/");
  const filename = slash >= 0 ? normalized.slice(slash + 1) : normalized;
  const subfolder = slash >= 0 ? normalized.slice(0, slash) : "";
  const params = new URLSearchParams({ filename, type });
  if (subfolder) params.set("subfolder", subfolder);
  return `/view?${params.toString()}`;
}

/** 把输出目录里的相对路径变成可载入的 URL（带时间戳避免缓存）。 */
export function outputFileUrl(name) {
  if (!name) return null;
  const params = new URLSearchParams({ filename: name, type: "output" });
  const slash = String(name).lastIndexOf("/");
  if (slash >= 0) {
    params.set("subfolder", String(name).slice(0, slash));
    params.set("filename", String(name).slice(slash + 1));
  }
  return `/view?${params.toString()}&t=${Date.now()}`;
}
