import { SceneManager } from './SceneManager';
import { HUDController } from './ui/hud';

const appRoot = document.getElementById('app')!;
const scene = new SceneManager(appRoot);
const hud = new HUDController(scene);

let last = performance.now();
function loop(now: number) {
  const dt = (now - last) / 1000;
  last = now;
  scene.update(dt);
  hud.tick();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

// Resize observer as a fallback for container size changes
const ro = new ResizeObserver(() => scene.onResize());
ro.observe(appRoot);

// Intro text in console (placeholder)
console.log('%cCosmic Heart', 'color:#7dd3fc; font-weight:bold;', '— 气息 · 星轨 · 同心');