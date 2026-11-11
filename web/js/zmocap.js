// zmocap 前端扩展：角色文件上传。
//
// ComfyUI 官方给图片/视频/音频提供了上传 widget（image_upload / video_upload / audio_upload），
// 但 3D 文件没有对应机制。这里复用官方上传接口 `POST /upload/image`
// （该接口不校验后缀，直接把文件存进 input/），上传成功后写进节点的下拉框并选中，
// 交互与官方 LoadImage 的「choose file to upload」一致。
//
// 骨骼画布预览在 zmocap_preview.js 里，别在这里重复注册。
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CHARACTER_NODES = new Set(["ZMocapVRMLoader"]);
const ACCEPT = ".vrm,.glb,.gltf,.VRM,.GLB,.GLTF";

async function uploadCharacter(file) {
    const body = new FormData();
    body.append("image", file, file.name);
    body.append("type", "input");
    body.append("overwrite", "true");
    const response = await api.fetchApi("/upload/image", { method: "POST", body });
    if (!response.ok) {
        throw new Error(`上传失败 (${response.status}): ${await response.text()}`);
    }
    const data = await response.json();
    return data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
}

function pushChoice(widget, name) {
    const bag = widget.options?.values ?? widget.values;
    if (Array.isArray(bag) && !bag.includes(name)) {
        bag.push(name);
    }
}

function addUploadButton(node) {
    const widget = node.widgets?.find((w) => w.name === "character");
    if (!widget || widget.__zmocapUpload) {
        return;
    }
    widget.__zmocapUpload = true;

    const button = node.addWidget("button", "upload vrm/glb...", null, () => {
        const picker = document.createElement("input");
        picker.type = "file";
        picker.accept = ACCEPT;
        picker.onchange = async () => {
            const file = picker.files?.[0];
            if (!file) {
                return;
            }
            try {
                const name = await uploadCharacter(file);
                pushChoice(widget, name);
                widget.value = name;
                widget.callback?.(name);
                node.setDirtyCanvas(true, true);
                console.log(`[zmocap] uploaded character: ${name}`);
            } catch (error) {
                console.error("[zmocap]", error);
                alert(String(error?.message || error));
            }
        };
        picker.click();
    });
    button.serialize = false;
}

app.registerExtension({
    name: "zmocap.upload",
    async setup() {
        console.log("[zmocap] upload extension loaded");
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!CHARACTER_NODES.has(nodeData.name)) {
            return;
        }
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            addUploadButton(this);
        };
    },
});
