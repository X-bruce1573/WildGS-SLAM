# Cosmic Heart — 最小可感知体验 (Vite + TS + Three.js)

一个可运行的告白体验雏形：
- 星空场景（Three.js）
- 亮点交互：哈气起雾 + 指尖擦拭（Canvas 2D）
- 节拍脉动：Tap 节拍驱动星空呼吸（Space 键/按钮）
- 占位音频：WebAudio 生成的轻柔环境底（首次点击开启）

## 本地运行
```bash
npm i
npm run dev
# 浏览器自动打开 http://localhost:5173
```

## 生产打包 & 预览
```bash
npm run build
npm run preview
# 打开 http://localhost:5174
```

## 自测清单
- 渲染：页面加载后能看到深色背景与三维星空，星点缓慢旋转。
- 雾面：点击“吹一口气”出现雾层；按住并拖动可擦拭露出星空；雾会缓慢消散。
- 节拍：点击“Tap 节拍”或按空格3次以上，HUD 显示 BPM；星空随节拍明暗脉动。
- 声音：点击“开启声音”后，听到轻微环境底，Tap 时会随拍轻微起伏。
- 移动端：
  - 按钮可点；
  - 可手指擦拭雾面；
  - 画面保持流畅（60fps 视设备）。

## 技术栈
- Vite + TypeScript
- Three.js（场景/相机/Points 星野）
- Canvas 2D（雾面与擦拭）
- WebAudio（占位环境底与随拍起伏）

## 结构
```
/public
/src
  /ui
    hud.ts
  SceneManager.ts
  main.ts
  styles.css
index.html
vite.config.ts
tsconfig.json
package.json
```

## 说明
- 没有外部音频文件，声音通过 WebAudio 合成，首交互后可播放。
- 可在此基础上扩展“连星成图”“同心对齐”等交互。