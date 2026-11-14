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

  const contactGroup = new THREE.Group();
  contactGroup.name = "zmocap_contact_rings";
  scene.add(contactGroup);

  const CONTACT_COLORS = { left: 0x3b82f6, right: 0xef4444 };

  /** 触地光圈：平铺在地面上，按帧开关，蓝=左、红=右。 */
  function makeRing(side) {
    const geometry = new THREE.RingGeometry(0.09, 0.135, 40);
    const material = new THREE.MeshBasicMaterial({
      color: CONTACT_COLORS[side],
      transparent: true,
      opacity: 0.9,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.name = `zmocap_contact_${side}`;
    mesh.rotation.x = -Math.PI / 2;
    mesh.visible = false;
    mesh.renderOrder = 3;
    contactGroup.add(mesh);
    return mesh;
  }
  const rings = { left: makeRing("left"), right: makeRing("right") };

  function updateContactRings() {
    if (!contact) return;
    const fps = contact.fps || 30;
    const total = contact.frames || 1;
    // 动画循环播放时 elapsed 会一直涨：按片段时长取模，否则只在第一轮里“有帧可查”。
    const looped = duration > 0 ? elapsed % duration : elapsed;
    const index = Math.min(total - 1, Math.max(0, Math.floor(looped * fps)));
    for (const side of ["left", "right"]) {
      const data = contact[side];
      const ring = rings[side];
      const position = data?.pos?.[index];
      const visible = contactVisible && Array.isArray(position);
      ring.visible = visible;
      if (visible) ring.position.set(position[0], position[1] + 0.002, position[2]);
    }
  }

  const grid = new THREE.GridHelper(6, 12, 0x445566, 0x223344);
  grid.material.opacity = 0.5;
  grid.material.transparent = true;
  scene.add(grid);

  let mixer = null;
  let current = null;
  let playing = true;
  let disposed = false;
  let frames = [];
  let elapsed = 0;
  let duration = 0;
  let contact = null;
  let contactVisible = true;

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
    if (mixer && playing) {
      mixer.update(delta);
      elapsed += delta;
    }
    updateContactRings();
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

  /**
   * 当前姿态的包围盒。
   *
   * `Box3.setFromObject()` 对 SkinnedMesh 取的是绑定姿态（SMPL-X 的 T-pose，
   * 脚在 y=-1.29），而动画会把根节点抬到世界地面 y=0；直接用绑定姿态会让
   * 角色看起来“飘在半空”。这里对蒙皮网格按骨骼采样当前姿态，普通网格/骨骼
   * 取世界位置，得到真正的地面。
   */
  function sceneBounds() {
    const box = new THREE.Box3();
    const vertex = new THREE.Vector3();
    const point = new THREE.Vector3();
    let found = false;
    current.updateMatrixWorld(true);
    // 有蒙皮网格时只按蒙皮网格算地面：小棍是显示层，可能比脚底更低。
    const skinned = [];
    current.traverse((object) => {
      if (object.isSkinnedMesh && object.skeleton
          && object.geometry?.attributes?.position
          && object.geometry.attributes.skinIndex
          && object.geometry.attributes.skinWeight) {
        skinned.push(object);
      }
    });
    if (skinned.length) {
      for (const object of skinned) {
        const position = object.geometry.attributes.position;
        const step = Math.max(1, Math.floor(position.count / 20000));
        for (let index = 0; index < position.count; index += step) {
          vertex.fromBufferAttribute(position, index);
          object.applyBoneTransform(index, vertex);
          vertex.applyMatrix4(object.matrixWorld);
          box.expandByPoint(vertex);
        }
        found = true;
      }
    }
    if (found) return box;
    current.traverse((object) => {
      if (object.isMesh) {
        const meshBox = new THREE.Box3().setFromObject(object);
        if (!meshBox.isEmpty()) {
          box.union(meshBox);
          found = true;
        }
      } else if (object.isBone) {
        box.expandByPoint(object.getWorldPosition(point));
        found = true;
      }
    });
    if (!found) box.setFromObject(current);
    return box;
  }

  function look(scaleHint) {
    const box = sceneBounds();
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
        duration = gltf.animations?.[0]?.duration || 0;
        if (gltf.animations?.length) {
          mixer = new THREE.AnimationMixer(current);
          for (const clip of gltf.animations) mixer.clipAction(clip).play();
          playing = autoplay;
          // 先应用第 0 帧，否则 sceneBounds() 看到的是 T-pose 绑定姿态。
          mixer.update(0);
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

    /** 载入触地数据（zmocap.contact.v1 的预览载荷）；null 表示清除光圈。 */
    /** 光圈显隐（前端开关，不参与工作流序列化）。 */
    setContactsVisible(visible) {
      contactVisible = !!visible;
      updateContactRings();
    },

    setContacts(payload) {
      contact = payload || null;
      if (!contact) {
        rings.left.visible = false;
        rings.right.visible = false;
      }
      updateContactRings();
      return !!contact;
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
      for (const side of ["left", "right"]) {
        rings[side].geometry.dispose();
        rings[side].material.dispose();
      }
      scene.remove(contactGroup);
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
