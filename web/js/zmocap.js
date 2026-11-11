// zmocap 前端扩展：后续用于 three.js 骨骼预览；当前仅注册版本信息。
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "zmocap.preview",
    async setup() {
        console.log("[zmocap] extension loaded (preview pipeline pending)");
    },
});
