import { SceneManager } from '../SceneManager';

export class HUDController {
  private fogCanvas: HTMLCanvasElement;
  private fogCtx: CanvasRenderingContext2D;
  private isErasing = false;
  private fogLevel = 0; // 0..1
  private fogFadeSpeed = 0.0;

  private btnSound: HTMLButtonElement;
  private btnFog: HTMLButtonElement;
  private btnTap: HTMLButtonElement;
  private bpmLabel: HTMLElement;

  private audioCtx: AudioContext | null = null;
  private gain: GainNode | null = null;
  private oscillators: OscillatorNode[] = [];

  private tapTimes: number[] = [];

  constructor(private scene: SceneManager) {
    this.fogCanvas = document.getElementById('fogCanvas') as HTMLCanvasElement;
    const ctx = this.fogCanvas.getContext('2d');
    if (!ctx) throw new Error('Cannot get 2D context for fog canvas');
    this.fogCtx = ctx;

    this.btnSound = document.getElementById('btnSound') as HTMLButtonElement;
    this.btnFog = document.getElementById('btnFog') as HTMLButtonElement;
    this.btnTap = document.getElementById('btnTap') as HTMLButtonElement;
    this.bpmLabel = document.getElementById('bpmLabel') as HTMLElement;

    this.layout();
    window.addEventListener('resize', () => this.layout());

    // Events
    this.fogCanvas.addEventListener('pointerdown', (e) => this.onPointerDown(e));
    window.addEventListener('pointermove', (e) => this.onPointerMove(e));
    window.addEventListener('pointerup', () => (this.isErasing = false));

    this.btnFog.addEventListener('click', () => this.puffFog());
    this.btnSound.addEventListener('click', () => this.enableAudio());
    this.btnTap.addEventListener('click', () => this.tap());
    window.addEventListener('keydown', (e) => {
      if (e.code === 'Space') {
        e.preventDefault();
        this.tap();
      }
    });

    // Initial gentle fog hint
    this.fogLevel = 0.2;
    this.drawFogImmediate();
    this.startFogFade(0.01);
  }

  private layout() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.fogCanvas.width = Math.floor(window.innerWidth * dpr);
    this.fogCanvas.height = Math.floor(window.innerHeight * dpr);
    this.fogCanvas.style.width = window.innerWidth + 'px';
    this.fogCanvas.style.height = window.innerHeight + 'px';
    this.drawFogImmediate();
  }

  private drawFogImmediate() {
    const ctx = this.fogCtx;
    const w = this.fogCanvas.width;
    const h = this.fogCanvas.height;

    ctx.globalCompositeOperation = 'source-over';
    ctx.clearRect(0, 0, w, h);

    if (this.fogLevel <= 0.001) return;

    const grad = ctx.createRadialGradient(w * 0.5, h * 0.55, Math.min(w, h) * 0.1, w * 0.5, h * 0.5, Math.max(w, h) * 0.7);
    grad.addColorStop(0, `rgba(180, 200, 220, ${0.35 * this.fogLevel})`);
    grad.addColorStop(1, `rgba(10, 15, 24, ${0.85 * this.fogLevel})`);

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  }

  private onPointerDown(e: PointerEvent) {
    this.isErasing = true;
    this.eraseAt(e);
  }

  private onPointerMove(e: PointerEvent) {
    if (!this.isErasing) return;
    this.eraseAt(e);
  }

  private eraseAt(e: PointerEvent) {
    if (this.fogLevel <= 0) return;
    const rect = this.fogCanvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const x = (e.clientX - rect.left) * dpr;
    const y = (e.clientY - rect.top) * dpr;

    const ctx = this.fogCtx;
    ctx.save();
    ctx.globalCompositeOperation = 'destination-out';
    const radius = Math.max(24, Math.min(this.fogCanvas.width, this.fogCanvas.height) * 0.06);
    const gradient = ctx.createRadialGradient(x, y, radius * 0.2, x, y, radius);
    gradient.addColorStop(0, 'rgba(0,0,0,0.9)');
    gradient.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  puffFog() {
    // Increase fog and start a gentle fade out
    this.fogLevel = Math.min(1, this.fogLevel + 0.6);
    this.drawFogImmediate();
    this.startFogFade(0.02);
  }

  private startFogFade(speed: number) {
    this.fogFadeSpeed = speed; // per frame reduction
  }

  enableAudio() {
    if (this.audioCtx) return;
    const ctx = new (window.AudioContext || (window as any).webkitAudioContext)();
    const gain = ctx.createGain();
    gain.gain.value = 0.0;
    gain.connect(ctx.destination);

    // Two gentle oscillators for airy pad
    const osc1 = ctx.createOscillator();
    osc1.type = 'sine';
    osc1.frequency.value = 220; // A3
    osc1.connect(gain);

    const osc2 = ctx.createOscillator();
    osc2.type = 'triangle';
    osc2.frequency.value = 440; // A4
    osc2.connect(gain);

    osc1.start();
    osc2.start();

    // Slow fade-in
    gain.gain.linearRampToValueAtTime(0.02, ctx.currentTime + 0.8);

    this.audioCtx = ctx;
    this.gain = gain;
    this.oscillators = [osc1, osc2];

    this.btnSound.disabled = true;
    this.btnSound.textContent = '声音已开启';
  }

  tap() {
    const now = performance.now();
    this.tapTimes.push(now);
    // Keep last 6 taps
    if (this.tapTimes.length > 6) this.tapTimes.shift();

    if (this.tapTimes.length >= 3) {
      const intervals: number[] = [];
      for (let i = 1; i < this.tapTimes.length; i++) {
        intervals.push(this.tapTimes[i] - this.tapTimes[i - 1]);
      }
      // Remove outliers by trimming
      intervals.sort((a, b) => a - b);
      const trimmed = intervals.slice(1, intervals.length - 1);
      const avg = (trimmed.length ? trimmed : intervals).reduce((a, b) => a + b, 0) / (trimmed.length ? trimmed.length : intervals.length);
      const bpm = 60000 / avg;

      this.scene.setHeartbeatFromBpm(bpm);
      this.bpmLabel.textContent = `BPM: ${Math.round(bpm)}`;

      // Optional: audio swell on beat
      if (this.audioCtx && this.gain) {
        const t = this.audioCtx.currentTime;
        const base = 0.018;
        this.gain.gain.cancelScheduledValues(t);
        this.gain.gain.setValueAtTime(base, t);
        this.gain.gain.linearRampToValueAtTime(base + 0.015, t + 0.05);
        this.gain.gain.linearRampToValueAtTime(base, t + 0.25);
      }
    }
  }

  // Called each frame from main loop
  tick() {
    if (this.fogLevel > 0 && this.fogFadeSpeed > 0) {
      this.fogLevel = Math.max(0, this.fogLevel - this.fogFadeSpeed);
      this.drawFogImmediate();
      if (this.fogLevel <= 0) {
        this.fogFadeSpeed = 0;
        const ctx = this.fogCtx;
        ctx.globalCompositeOperation = 'source-over';
        ctx.clearRect(0, 0, this.fogCanvas.width, this.fogCanvas.height);
      }
    }
  }
}