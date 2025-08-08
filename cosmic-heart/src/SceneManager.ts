import * as THREE from 'three';

export class SceneManager {
  private container: HTMLElement;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private stars: THREE.Points<THREE.BufferGeometry, THREE.PointsMaterial> | null = null;
  private starMaterial: THREE.PointsMaterial | null = null;
  private lastWidth = 0;
  private lastHeight = 0;

  // Heartbeat control (ms per beat)
  private heartbeatMs = 800; // default ~75 BPM
  private time = 0;

  constructor(container: HTMLElement) {
    this.container = container;

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x06070b, 0.015);

    this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 2000);
    this.camera.position.set(0, 0, 120);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.container.appendChild(this.renderer.domElement);

    const ambient = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambient);

    this.addStars();

    this.onResize();
    window.addEventListener('resize', () => this.onResize());
  }

  private addStars() {
    const starCount = 3000;
    const positions = new Float32Array(starCount * 3);
    const color = new THREE.Color(0xb8d8ff);

    // Distribute stars in a sphere shell
    for (let i = 0; i < starCount; i++) {
      const r = 400 * Math.cbrt(Math.random());
      const theta = Math.acos(THREE.MathUtils.randFloatSpread(2));
      const phi = THREE.MathUtils.randFloat(0, Math.PI * 2);
      const x = r * Math.sin(theta) * Math.cos(phi);
      const y = r * Math.sin(theta) * Math.sin(phi);
      const z = r * Math.cos(theta);
      positions[i * 3 + 0] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    this.starMaterial = new THREE.PointsMaterial({
      size: 1.8,
      sizeAttenuation: true,
      color,
      transparent: true,
      depthWrite: false,
      blending: THREE.AdditiveBlending
    });

    this.stars = new THREE.Points(geometry, this.starMaterial);
    this.scene.add(this.stars);
  }

  setHeartbeatFromBpm(bpm: number) {
    const clamped = THREE.MathUtils.clamp(bpm, 40, 200);
    this.heartbeatMs = 60000 / clamped;
  }

  update(dt: number) {
    this.time += dt;

    // Subtle starfield rotation for depth
    if (this.stars) {
      this.stars.rotation.y += dt * 0.01;
      this.stars.rotation.x += dt * 0.005;
    }

    // Heartbeat brightness pulse: ease-in quick, ease-out slow
    if (this.starMaterial) {
      const t = (this.time * 1000) % this.heartbeatMs; // in ms
      const phase = t / this.heartbeatMs; // 0..1

      // double-beat: quick pulse then echo
      const beat = (p: number) => {
        const sharp = Math.max(0, 1 - Math.abs((p - 0.05) / 0.05));
        const echo = Math.max(0, 1 - Math.abs((p - 0.3) / 0.09));
        return sharp * 1.0 + echo * 0.5;
      };

      const brightness = 0.35 + 0.65 * Math.min(1, beat(phase));
      const baseColor = new THREE.Color(0x9cc7ff);
      const final = baseColor.clone().multiplyScalar(brightness);
      this.starMaterial.color.copy(final);
      this.starMaterial.needsUpdate = true;
    }

    this.renderer.render(this.scene, this.camera);
  }

  onResize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (w === this.lastWidth && h === this.lastHeight) return;

    this.lastWidth = w;
    this.lastHeight = h;

    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();

    this.renderer.setSize(w, h, false);
  }
}