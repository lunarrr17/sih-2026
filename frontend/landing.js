/* ========================================
   IP-SAKTI SAHAYAK — LANDING PAGE JS
   Ancient Wisdom × Digital Intelligence
   ======================================== */

'use strict';

// ─────────────────────────────────────────
//  GLOBALS
// ─────────────────────────────────────────
const mouse = { x: 0, y: 0, nx: 0, ny: 0 };
let scrollY = 0;
let heroHeight = 0;
let rafId = null;

// Color palette
const PALETTE = {
  deepGreen:  0x12372A,
  botGreen:   0x2E7D32,
  emGreen:    0x3E7C59,
  sageGreen:  0xA8C686,
  mint:       0xDDEED5,
  ivory:      0xF7F4E8,
  beige:      0xE9DFC5,
  gold:       0xBFA76F,
};

// ─────────────────────────────────────────
//  SCROLL PROGRESS BAR
// ─────────────────────────────────────────
const progressBar = document.createElement('div');
progressBar.className = 'scroll-progress';
document.body.prepend(progressBar);

function updateScrollProgress() {
  const docH = document.documentElement.scrollHeight - window.innerHeight;
  const pct  = docH > 0 ? (window.scrollY / docH) * 100 : 0;
  progressBar.style.width = pct + '%';
}

// ─────────────────────────────────────────
//  NAVBAR SCROLL EFFECT
// ─────────────────────────────────────────
const navbar = document.getElementById('navbar');

function handleNavbarScroll() {
  if (window.scrollY > 60) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
}

// ─────────────────────────────────────────
//  FLOATING LEAVES (CSS-Animated)
// ─────────────────────────────────────────
const leavesContainer = document.getElementById('leavesContainer');

const LEAF_COLORS = ['#3E7C59', '#2E7D32', '#A8C686', '#DDEED5', '#12372A', '#BFA76F'];
const LEAF_TYPES  = [leafSVG1, leafSVG2, leafSVG3, leafSVG4];

function leafSVG1(color, size) {
  return `<svg width="${size}" height="${size * 0.7}" viewBox="0 0 60 40" fill="none">
    <path d="M5 35 Q30 5 55 15 Q45 38 5 35Z" fill="${color}" opacity="0.75"/>
    <path d="M5 35 Q30 20 55 15" stroke="${color}" stroke-width="1.2" fill="none" opacity="0.4"/>
  </svg>`;
}

function leafSVG2(color, size) {
  return `<svg width="${size * 0.6}" height="${size}" viewBox="0 0 36 60" fill="none">
    <path d="M18 55 Q5 30 18 5 Q31 30 18 55Z" fill="${color}" opacity="0.7"/>
    <path d="M18 55 L18 5" stroke="${color}" stroke-width="1" fill="none" opacity="0.35"/>
  </svg>`;
}

function leafSVG3(color, size) {
  return `<svg width="${size}" height="${size * 0.65}" viewBox="0 0 70 45" fill="none">
    <path d="M3 40 Q20 3 50 8 Q65 25 55 42 Q35 50 3 40Z" fill="${color}" opacity="0.72"/>
    <path d="M3 40 Q35 15 55 42" stroke="${color}" stroke-width="1" fill="none" opacity="0.35"/>
  </svg>`;
}

function leafSVG4(color, size) {
  return `<svg width="${size * 0.55}" height="${size}" viewBox="0 0 33 60" fill="none">
    <path d="M16 58 Q2 35 16 2 Q30 35 16 58Z" fill="${color}" opacity="0.65"/>
    <path d="M16 2 Q8 30 16 58" stroke="${color}" stroke-width="0.8" fill="none" opacity="0.3"/>
    <path d="M16 20 Q8 25 4 38" stroke="${color}" stroke-width="0.7" fill="none" opacity="0.25"/>
    <path d="M16 25 Q24 30 28 42" stroke="${color}" stroke-width="0.7" fill="none" opacity="0.25"/>
  </svg>`;
}

const LEAF_COUNT_DESKTOP = 28;
const LEAF_COUNT_MOBILE  = 14;

function createLeaves() {
  if (!leavesContainer) return;
  leavesContainer.innerHTML = '';
  const count = window.innerWidth < 768 ? LEAF_COUNT_MOBILE : LEAF_COUNT_DESKTOP;

  for (let i = 0; i < count; i++) {
    const el    = document.createElement('div');
    el.className = 'leaf-item';

    const color  = LEAF_COLORS[Math.floor(Math.random() * LEAF_COLORS.length)];
    const size   = 18 + Math.random() * 36;
    const type   = LEAF_TYPES[Math.floor(Math.random() * LEAF_TYPES.length)];
    const startX = Math.random() * 100;
    const startY = Math.random() * 100;
    const layer  = Math.random(); // 0 = bg, 1 = fg
    const dur    = 12 + Math.random() * 20;
    const delay  = Math.random() * -20;
    const drift  = (Math.random() - 0.5) * 120;
    const rot    = Math.random() * 360;
    const rotEnd = rot + (Math.random() - 0.5) * 180;
    const zIndex = Math.floor(layer * 6) + 2;
    const opacity = 0.35 + layer * 0.55;

    el.innerHTML = type(color, size);
    el.style.cssText = `
      left: ${startX}%;
      top: ${startY}%;
      z-index: ${zIndex};
      opacity: ${opacity};
      transform: rotate(${rot}deg);
      animation: leafFloat${i} ${dur}s ${delay}s ease-in-out infinite alternate;
      will-change: transform;
    `;

    // Inject a unique keyframe for each leaf
    const style = document.createElement('style');
    style.textContent = `
      @keyframes leafFloat${i} {
        0%   { transform: translate(0, 0) rotate(${rot}deg) scale(1); }
        33%  { transform: translate(${drift * 0.4}px, ${-30 - Math.random() * 40}px) rotate(${rot + 40}deg) scale(${0.9 + Math.random() * 0.2}); }
        66%  { transform: translate(${drift * 0.8}px, ${-50 - Math.random() * 30}px) rotate(${rot + 80}deg) scale(${0.85 + Math.random() * 0.15}); }
        100% { transform: translate(${drift}px, ${-70 - Math.random() * 30}px) rotate(${rotEnd}deg) scale(0.8); }
      }
    `;
    document.head.appendChild(style);

    leavesContainer.appendChild(el);
  }
}

// Mouse parallax for leaves
function updateLeafParallax() {
  if (!leavesContainer) return;
  const leaves = leavesContainer.querySelectorAll('.leaf-item');
  leaves.forEach((leaf, i) => {
    const factor = ((i % 5) + 1) * 0.008;
    const px = (mouse.nx - 0.5) * -factor * 60;
    const py = (mouse.ny - 0.5) * -factor * 40;
    leaf.style.transform += ` translate(${px}px, ${py}px)`;
  });
}

// ─────────────────────────────────────────
//  THREE.JS HERO PARTICLE SYSTEM
// ─────────────────────────────────────────
let threeScene, threeCamera, threeRenderer, particlesMesh, particlePositions;
let threeAnimating = false;

function initThreeHero() {
  const canvas = document.getElementById('heroCanvas');
  if (!canvas || typeof THREE === 'undefined') return;

  const W = canvas.parentElement.clientWidth;
  const H = canvas.parentElement.clientHeight;

  // Scene
  threeScene = new THREE.Scene();
  threeCamera = new THREE.PerspectiveCamera(60, W / H, 0.1, 1000);
  threeCamera.position.set(0, 0, 50);

  threeRenderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  threeRenderer.setSize(W, H);
  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  threeRenderer.setClearColor(0x000000, 0);

  // Particles
  const COUNT = window.innerWidth < 768 ? 600 : 1400;
  const geometry = new THREE.BufferGeometry();
  particlePositions = new Float32Array(COUNT * 3);
  const colors = new Float32Array(COUNT * 3);

  const colorOptions = [
    new THREE.Color(PALETTE.sageGreen),
    new THREE.Color(PALETTE.emGreen),
    new THREE.Color(PALETTE.mint),
    new THREE.Color(PALETTE.gold),
    new THREE.Color(PALETTE.botGreen),
  ];

  for (let i = 0; i < COUNT; i++) {
    const r = 20 + Math.random() * 55;
    const theta = Math.random() * Math.PI * 2;
    const phi   = Math.acos(2 * Math.random() - 1);
    particlePositions[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
    particlePositions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    particlePositions[i * 3 + 2] = r * Math.cos(phi) - 10;

    const col = colorOptions[Math.floor(Math.random() * colorOptions.length)];
    colors[i * 3]     = col.r;
    colors[i * 3 + 1] = col.g;
    colors[i * 3 + 2] = col.b;
  }

  geometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
  geometry.setAttribute('color',    new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 0.55,
    vertexColors: true,
    transparent: true,
    opacity: 0.7,
    sizeAttenuation: true,
  });

  particlesMesh = new THREE.Points(geometry, material);
  threeScene.add(particlesMesh);

  // Ambient light
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
  threeScene.add(ambientLight);

  threeAnimating = true;
  animateThree();

  // Resize
  window.addEventListener('resize', () => {
    const nW = canvas.parentElement.clientWidth;
    const nH = canvas.parentElement.clientHeight;
    threeCamera.aspect = nW / nH;
    threeCamera.updateProjectionMatrix();
    threeRenderer.setSize(nW, nH);
  });
}

let threeFrame = 0;

function animateThree() {
  if (!threeAnimating) return;
  requestAnimationFrame(animateThree);
  threeFrame += 0.004;

  if (particlesMesh) {
    particlesMesh.rotation.y = threeFrame * 0.15 + mouse.nx * 0.3;
    particlesMesh.rotation.x = Math.sin(threeFrame * 0.2) * 0.1 + mouse.ny * 0.15;

    // Subtle wave on particle positions
    const pos = particlesMesh.geometry.attributes.position.array;
    for (let i = 0; i < pos.length; i += 9) {
      pos[i + 1] = particlePositions[i + 1] + Math.sin(threeFrame + i * 0.01) * 0.3;
    }
    particlesMesh.geometry.attributes.position.needsUpdate = true;
  }

  if (threeCamera) {
    threeCamera.position.x += ((mouse.nx - 0.5) * 8 - threeCamera.position.x) * 0.02;
    threeCamera.position.y += ((0.5 - mouse.ny) * 5 - threeCamera.position.y) * 0.02;
    threeCamera.lookAt(0, 0, 0);
  }

  threeRenderer.render(threeScene, threeCamera);
}

// ─────────────────────────────────────────
//  NEURAL NETWORK CANVAS (2D)
// ─────────────────────────────────────────
let neuralCanvas, neuralCtx;
let neuralNodes = [];
let neuralParticles = [];

function initNeuralCanvas() {
  neuralCanvas = document.getElementById('neuralCanvas');
  if (!neuralCanvas) return;
  neuralCtx = neuralCanvas.getContext('2d');

  resizeNeuralCanvas();

  // Create organic nodes scattered around hero (right side)
  const W = neuralCanvas.width;
  const H = neuralCanvas.height;

  for (let i = 0; i < 28; i++) {
    neuralNodes.push({
      x: W * 0.45 + Math.random() * W * 0.55,
      y: H * 0.1  + Math.random() * H * 0.85,
      ox: 0, oy: 0,
      r: 1.5 + Math.random() * 3,
      speed: 0.002 + Math.random() * 0.006,
      phase: Math.random() * Math.PI * 2,
      pulseSpeed: 0.5 + Math.random() * 1.5,
      color: [
        'rgba(62,124,89,',
        'rgba(168,198,134,',
        'rgba(191,167,111,',
        'rgba(221,238,213,',
      ][Math.floor(Math.random() * 4)],
    });
  }

  // Travelling particles along edges
  for (let i = 0; i < 10; i++) {
    neuralParticles.push({
      progress: Math.random(),
      speed: 0.003 + Math.random() * 0.006,
      fromIdx: Math.floor(Math.random() * neuralNodes.length),
      toIdx: Math.floor(Math.random() * neuralNodes.length),
      size: 2 + Math.random() * 2,
    });
  }

  window.addEventListener('resize', resizeNeuralCanvas);
}

function resizeNeuralCanvas() {
  if (!neuralCanvas) return;
  const parent = neuralCanvas.parentElement;
  neuralCanvas.width  = parent.clientWidth;
  neuralCanvas.height = parent.clientHeight;
}

function drawNeuralNetwork(timestamp) {
  if (!neuralCtx) return;
  const W = neuralCanvas.width;
  const H = neuralCanvas.height;

  neuralCtx.clearRect(0, 0, W, H);

  const t = timestamp * 0.001;

  // Update node floating positions
  neuralNodes.forEach(n => {
    n.ox = Math.sin(t * n.speed * 80 + n.phase) * 12;
    n.oy = Math.cos(t * n.speed * 60 + n.phase) * 10;

    // Subtle mouse interaction
    const dx = n.x - mouse.nx * W;
    const dy = n.y - mouse.ny * H;
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 160) {
      const factor = (160 - dist) / 160 * 0.04;
      n.ox += dx * factor;
      n.oy += dy * factor;
    }
  });

  // Draw organic connections (only nearby nodes)
  for (let i = 0; i < neuralNodes.length; i++) {
    for (let j = i + 1; j < neuralNodes.length; j++) {
      const a  = neuralNodes[i];
      const b  = neuralNodes[j];
      const ax = a.x + a.ox;
      const ay = a.y + a.oy;
      const bx = b.x + b.ox;
      const by = b.y + b.oy;
      const dx = ax - bx;
      const dy = ay - by;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < 180) {
        const alpha = (1 - dist / 180) * 0.22;
        const grad = neuralCtx.createLinearGradient(ax, ay, bx, by);
        grad.addColorStop(0, a.color + alpha + ')');
        grad.addColorStop(1, b.color + alpha + ')');
        neuralCtx.beginPath();
        neuralCtx.moveTo(ax, ay);

        // Organic curve with slight arc
        const mx = (ax + bx) / 2 + (Math.random() - 0.5) * 20;
        const my = (ay + by) / 2 + (Math.random() - 0.5) * 20;
        neuralCtx.quadraticCurveTo(mx, my, bx, by);

        neuralCtx.strokeStyle = grad;
        neuralCtx.lineWidth = 0.8;
        neuralCtx.stroke();
      }
    }
  }

  // Draw node circles
  neuralNodes.forEach((n, i) => {
    const nx = n.x + n.ox;
    const ny = n.y + n.oy;
    const pulse = 1 + Math.sin(t * n.pulseSpeed + n.phase) * 0.3;
    const r = n.r * pulse;

    // Glow
    const grd = neuralCtx.createRadialGradient(nx, ny, 0, nx, ny, r * 4);
    grd.addColorStop(0, n.color + '0.4)');
    grd.addColorStop(1, n.color + '0)');
    neuralCtx.beginPath();
    neuralCtx.arc(nx, ny, r * 4, 0, Math.PI * 2);
    neuralCtx.fillStyle = grd;
    neuralCtx.fill();

    // Core
    neuralCtx.beginPath();
    neuralCtx.arc(nx, ny, r, 0, Math.PI * 2);
    neuralCtx.fillStyle = n.color + '0.85)';
    neuralCtx.fill();
  });

  // Travelling particles
  neuralParticles.forEach(p => {
    p.progress += p.speed;
    if (p.progress >= 1) {
      p.progress = 0;
      p.fromIdx = p.toIdx;
      p.toIdx = Math.floor(Math.random() * neuralNodes.length);
    }

    const from = neuralNodes[p.fromIdx];
    const to   = neuralNodes[p.toIdx];
    const px = from.x + (to.x - from.x) * p.progress + from.ox * (1 - p.progress) + to.ox * p.progress;
    const py = from.y + (to.y - from.y) * p.progress + from.oy * (1 - p.progress) + to.oy * p.progress;

    const grd = neuralCtx.createRadialGradient(px, py, 0, px, py, p.size * 3);
    grd.addColorStop(0, 'rgba(191,167,111,0.9)');
    grd.addColorStop(1, 'rgba(191,167,111,0)');
    neuralCtx.beginPath();
    neuralCtx.arc(px, py, p.size * 3, 0, Math.PI * 2);
    neuralCtx.fillStyle = grd;
    neuralCtx.fill();

    neuralCtx.beginPath();
    neuralCtx.arc(px, py, p.size, 0, Math.PI * 2);
    neuralCtx.fillStyle = 'rgba(191,167,111,0.95)';
    neuralCtx.fill();
  });
}

// ─────────────────────────────────────────
//  ECOSYSTEM CANVAS (2D organic connections)
// ─────────────────────────────────────────
let ecoCanvas, ecoCtx;

function initEcoCanvas() {
  ecoCanvas = document.getElementById('ecoCanvas');
  if (!ecoCanvas) return;
  ecoCtx = ecoCanvas.getContext('2d');

  const viz = document.getElementById('ecosystemViz');
  if (viz) {
    ecoCanvas.width  = viz.clientWidth;
    ecoCanvas.height = viz.clientHeight;
  }
}

function drawEcoConnections(timestamp) {
  if (!ecoCtx || !ecoCanvas) return;
  const W = ecoCanvas.width;
  const H = ecoCanvas.height;

  ecoCtx.clearRect(0, 0, W, H);

  const cx = W / 2;
  const cy = H / 2;
  const radius = Math.min(W, H) * 0.38;
  const t = timestamp * 0.001;
  const nodeCount = 6;

  for (let i = 0; i < nodeCount; i++) {
    const angle = (i / nodeCount) * Math.PI * 2 + t * 0.05;
    const nx = cx + Math.cos(angle) * radius;
    const ny = cy + Math.sin(angle) * radius;

    // Root-like connection from center to node
    const cp1x = cx + (nx - cx) * 0.4 + Math.sin(t * 0.3 + i) * 20;
    const cp1y = cy + (ny - cy) * 0.4 + Math.cos(t * 0.4 + i) * 15;
    const cp2x = cx + (nx - cx) * 0.7 + Math.sin(t * 0.25 + i) * 15;
    const cp2y = cy + (ny - cy) * 0.7 + Math.cos(t * 0.35 + i) * 10;

    const alpha = 0.18 + Math.sin(t * 0.6 + i * 0.8) * 0.08;
    ecoCtx.beginPath();
    ecoCtx.moveTo(cx, cy);
    ecoCtx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, nx, ny);
    ecoCtx.strokeStyle = `rgba(62,124,89,${alpha})`;
    ecoCtx.lineWidth = 1.5;
    ecoCtx.stroke();

    // Small travelling dot along the path
    const prog = (Math.sin(t * 0.8 + i * 1.1) + 1) / 2;
    const tx = bezierPoint(cx, cp1x, cp2x, nx, prog);
    const ty = bezierPoint(cy, cp1y, cp2y, ny, prog);

    const dotGrd = ecoCtx.createRadialGradient(tx, ty, 0, tx, ty, 5);
    dotGrd.addColorStop(0, 'rgba(191,167,111,0.9)');
    dotGrd.addColorStop(1, 'rgba(191,167,111,0)');
    ecoCtx.beginPath();
    ecoCtx.arc(tx, ty, 5, 0, Math.PI * 2);
    ecoCtx.fillStyle = dotGrd;
    ecoCtx.fill();
  }
}

function bezierPoint(p0, p1, p2, p3, t) {
  const mt = 1 - t;
  return mt * mt * mt * p0 + 3 * mt * mt * t * p1 + 3 * mt * t * t * p2 + t * t * t * p3;
}

// ─────────────────────────────────────────
//  BOOK PARALLAX ON MOUSE
// ─────────────────────────────────────────
function updateBookParallax() {
  const book = document.getElementById('book3D');
  const bookWrapper = document.getElementById('bookWrapper');
  const heroContent = document.getElementById('heroContent');

  if (book) {
    const rx = (mouse.ny - 0.5) * 12 + 8;
    const ry = (mouse.nx - 0.5) * -18 - 8;
    book.style.transform = `rotateX(${rx}deg) rotateY(${ry}deg)`;
  }

  if (bookWrapper) {
    const tx = (mouse.nx - 0.5) * -15;
    const ty = (mouse.ny - 0.5) * -10;
    bookWrapper.style.transform = `translate(${tx}px, ${ty}px)`;
  }

  if (heroContent) {
    const tx = (mouse.nx - 0.5) * 6;
    const ty = (mouse.ny - 0.5) * 4;
    heroContent.style.transform = `translate(${tx}px, ${ty}px)`;
  }
}

// ─────────────────────────────────────────
//  HEADLINE PARALLAX
// ─────────────────────────────────────────
function updateHeadlineParallax() {
  const lines = document.querySelectorAll('.headline-line[data-parallax]');
  lines.forEach(line => {
    const factor = parseFloat(line.dataset.parallax);
    const tx = (mouse.nx - 0.5) * factor * -60;
    const ty = (mouse.ny - 0.5) * factor * -30;
    line.style.transform = `translate(${tx}px, ${ty}px)`;
  });
}

// ─────────────────────────────────────────
//  SCROLL REVEAL (Intersection Observer)
// ─────────────────────────────────────────
function initScrollReveal() {
  const elements = document.querySelectorAll('.reveal-up, .reveal-right, .reveal-left');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const delay = parseFloat(entry.target.style.getPropertyValue('--delay') || '0') * 1000;
        setTimeout(() => {
          entry.target.classList.add('revealed');
        }, delay);
        observer.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15 });

  elements.forEach(el => observer.observe(el));
}

// ─────────────────────────────────────────
//  ROOT TRANSITION ANIMATION
// ─────────────────────────────────────────
function initRootTransitions() {
  const roots = document.querySelectorAll('.tr-root');

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        roots.forEach((r, i) => {
          r.style.transition = `stroke-dashoffset ${1.5 + i * 0.3}s ease ${i * 0.2}s`;
          r.style.strokeDashoffset = '0';
        });
        observer.disconnect();
      }
    });
  }, { threshold: 0.1 });

  const section = document.querySelector('.section-roots-transition');
  if (section) observer.observe(section);
}

// ─────────────────────────────────────────
//  SMOOTH MOUSE TRACKING
// ─────────────────────────────────────────
function initMouseTracking() {
  window.addEventListener('mousemove', (e) => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.nx = e.clientX / window.innerWidth;
    mouse.ny = e.clientY / window.innerHeight;
  });

  // Touch support
  window.addEventListener('touchmove', (e) => {
    if (e.touches.length > 0) {
      mouse.x  = e.touches[0].clientX;
      mouse.y  = e.touches[0].clientY;
      mouse.nx = e.touches[0].clientX / window.innerWidth;
      mouse.ny = e.touches[0].clientY / window.innerHeight;
    }
  }, { passive: true });
}

// ─────────────────────────────────────────
//  HERO SCROLL PARALLAX
// ─────────────────────────────────────────
function handleHeroScroll() {
  scrollY = window.scrollY;
  heroHeight = document.getElementById('hero')?.offsetHeight || window.innerHeight;
  const progress = Math.min(scrollY / heroHeight, 1);

  const bookContainer = document.getElementById('bookContainer');
  if (bookContainer) {
    bookContainer.style.transform = `translateY(${progress * 80}px)`;
    bookContainer.style.opacity = `${1 - progress * 1.2}`;
  }

  const heroContent = document.getElementById('heroContent');
  if (heroContent && scrollY > 0) {
    heroContent.style.transform = `translateY(${progress * 60}px)`;
  }

  const scrollHint = document.getElementById('scrollHint');
  if (scrollHint) {
    scrollHint.style.opacity = `${1 - progress * 5}`;
  }
}

// ─────────────────────────────────────────
//  MAIN RENDER LOOP
// ─────────────────────────────────────────
let lastTimestamp = 0;

function mainLoop(timestamp) {
  const dt = timestamp - lastTimestamp;
  lastTimestamp = timestamp;

  updateBookParallax();
  updateHeadlineParallax();
  drawNeuralNetwork(timestamp);
  drawEcoConnections(timestamp);

  rafId = requestAnimationFrame(mainLoop);
}

// ─────────────────────────────────────────
//  NAVBAR HAMBURGER MENU
// ─────────────────────────────────────────
function initHamburger() {
  const btn   = document.getElementById('navHamburger');
  const links = document.querySelector('.nav-links');
  if (!btn || !links) return;

  btn.addEventListener('click', () => {
    const isOpen = links.classList.toggle('mobile-open');
    btn.classList.toggle('is-open', isOpen);
    links.style.display = isOpen ? 'flex' : '';
    if (isOpen) {
      links.style.flexDirection = 'column';
      links.style.position = 'absolute';
      links.style.top = '72px';
      links.style.left = '0';
      links.style.right = '0';
      links.style.background = 'rgba(247, 244, 232, 0.97)';
      links.style.padding = '24px 40px';
      links.style.backdropFilter = 'blur(20px)';
      links.style.borderBottom = '1px solid rgba(191, 167, 111, 0.2)';
      links.style.zIndex = '999';
      links.style.gap = '24px';
    } else {
      links.style.cssText = '';
    }
  });
}

// ─────────────────────────────────────────
//  HERO BUTTON PARTICLE EFFECT
// ─────────────────────────────────────────
function initButtonEffects() {
  const btn = document.getElementById('btnExplore');
  if (!btn) return;

  btn.addEventListener('mouseenter', () => {
    for (let i = 0; i < 8; i++) {
      const p = document.createElement('div');
      p.style.cssText = `
        position: absolute;
        width: ${4 + Math.random() * 6}px;
        height: ${4 + Math.random() * 6}px;
        border-radius: 50%;
        background: rgba(168,198,134,0.8);
        left: ${20 + Math.random() * 60}%;
        top: ${20 + Math.random() * 60}%;
        pointer-events: none;
        z-index: 20;
        animation: particleBurst 0.7s ease forwards;
      `;
      btn.appendChild(p);
      setTimeout(() => p.remove(), 700);
    }
  });

  // Inject burst keyframe
  const s = document.createElement('style');
  s.textContent = `
    @keyframes particleBurst {
      from { opacity: 1; transform: translate(0, 0) scale(1); }
      to   { opacity: 0; transform: translate(${(Math.random()-0.5)*40}px, ${-20 - Math.random()*20}px) scale(0.3); }
    }
  `;
  document.head.appendChild(s);
}

// ─────────────────────────────────────────
//  SMOOTH SCROLL FOR ANCHOR LINKS
// ─────────────────────────────────────────
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
}

// ─────────────────────────────────────────
//  STATS COUNTER ANIMATION
// ─────────────────────────────────────────
function animateCounters() {
  const stats = document.querySelectorAll('.stat-num');
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const el = entry.target;
        const text = el.textContent;
        const num  = parseFloat(text.replace(/[^0-9.]/g, ''));
        const suffix = text.replace(/[0-9.]/g, '');
        const dur = 1800;
        const start = performance.now();

        function update(now) {
          const progress = Math.min((now - start) / dur, 1);
          const eased = 1 - Math.pow(1 - progress, 3); // ease-out-cubic
          const current = (num * eased).toFixed(num < 10 ? 1 : 0);
          el.textContent = current + suffix;
          if (progress < 1) requestAnimationFrame(update);
        }

        requestAnimationFrame(update);
        observer.unobserve(el);
      }
    });
  }, { threshold: 0.5 });

  stats.forEach(s => observer.observe(s));
}

// ─────────────────────────────────────────
//  INIT ALL
// ─────────────────────────────────────────
// ─────────────────────────────────────────
//  BACKEND CONFIG
// ─────────────────────────────────────────
const API_BASE_LANDING = 'http://127.0.0.1:8000';

// ─────────────────────────────────────────
//  HEALTH CHECK — Status Badge in Navbar
// ─────────────────────────────────────────
async function checkBackendHealth() {
  const dot   = document.getElementById('statusDot');
  const label = document.getElementById('statusLabel');
  const bkSt  = document.getElementById('backendStatus');

  try {
    const res  = await fetch(`${API_BASE_LANDING}/health`, { signal: AbortSignal.timeout(4000) });
    if (res.ok) {
      const data = await res.json();
      if (dot)   { dot.style.background = '#4ade80'; dot.style.boxShadow = '0 0 6px #4ade80'; }
      if (label) label.textContent = '● Online';
      if (bkSt)  bkSt.textContent = `✅ ${data.service || 'API'} v${data.version || '1.0'} Online`;
    } else {
      throw new Error('non-ok');
    }
  } catch {
    if (dot)   { dot.style.background = '#f87171'; dot.style.boxShadow = '0 0 6px #f87171'; }
    if (label) label.textContent = '● Offline';
    if (bkSt)  bkSt.textContent = '⚠️ Backend Offline — Start uvicorn';
  }
}

// ─────────────────────────────────────────
//  LANDING LIVE DEMO CHAT
// ─────────────────────────────────────────
async function landingAsk(prefill) {
  const input   = document.getElementById('demoInput');
  const chatArea = document.getElementById('demoChatArea');
  const sendBtn  = document.getElementById('demoSendBtn');

  const query = prefill || (input ? input.value.trim() : '');
  if (!query) return;

  if (input) input.value = '';

  // User bubble
  const userMsg = document.createElement('div');
  userMsg.className = 'demo-msg user-demo';
  userMsg.innerHTML = `<div class="demo-bubble user-bubble"><p>${escapeHtml(query)}</p></div><span class="demo-avatar">👤</span>`;
  chatArea.appendChild(userMsg);

  // Loading bubble
  const loadMsg = document.createElement('div');
  loadMsg.className = 'demo-msg bot-demo';
  loadMsg.innerHTML = `<span class="demo-avatar">🌿</span><div class="demo-bubble demo-loading"><span class="demo-dots"><span></span><span></span><span></span></span></div>`;
  chatArea.appendChild(loadMsg);
  chatArea.scrollTop = chatArea.scrollHeight;

  if (sendBtn) sendBtn.disabled = true;

  try {
    const res = await fetch(`${API_BASE_LANDING}/api/chat/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, jurisdiction: 'national', session_id: 'landing_demo' })
    });

    const data = await res.json();
    loadMsg.remove();

    const confidence = Math.round((data.confidence_score || 0.85) * 100);
    const botMsg = document.createElement('div');
    botMsg.className = 'demo-msg bot-demo';
    botMsg.innerHTML = `
      <span class="demo-avatar">🌿</span>
      <div class="demo-bubble">
        <div class="demo-confidence">🎯 ${confidence}% Grounded</div>
        <div class="demo-answer">${landingFormatMd(data.answer || '')}</div>
        ${data.citations && data.citations.length > 0 ? `
          <div class="demo-citations">
            ${data.citations.slice(0, 3).map(c =>
              `<span class="demo-cite-pill">📖 ${escapeHtml(c.title || c.statute || 'Source')}</span>`
            ).join('')}
          </div>` : ''}
        <a href="index.html" class="demo-full-link">Open full workspace for more →</a>
      </div>
    `;
    chatArea.appendChild(botMsg);
  } catch (err) {
    loadMsg.remove();
    const errMsg = document.createElement('div');
    errMsg.className = 'demo-msg bot-demo';
    errMsg.innerHTML = `<span class="demo-avatar">🌿</span><div class="demo-bubble demo-error"><p>⚠️ Backend not reachable. <a href="index.html">Open the full app</a> after starting the server.</p></div>`;
    chatArea.appendChild(errMsg);
  }

  if (sendBtn) sendBtn.disabled = false;
  chatArea.scrollTop = chatArea.scrollHeight;
}

function escapeHtml(str) {
  return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

function landingFormatMd(text) {
  return text
    .replace(/^### (.*$)/gim, '<strong style="color:#3E7C59;">$1</strong><br/>')
    .replace(/^## (.*$)/gim, '<strong style="color:#2E7D32; font-size:1.05em;">$1</strong><br/>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code style="background:rgba(62,124,89,0.12);padding:1px 5px;border-radius:4px;font-size:0.9em;">$1</code>')
    .replace(/^- (.*$)/gim, '<li style="margin-left:16px;">$1</li>')
    .replace(/\n\n/g, '<br/><br/>')
    .replace(/\n/g, '<br/>');
}

// Enter key support for demo input
document.addEventListener('DOMContentLoaded', () => {
  const demoInput = document.getElementById('demoInput');
  if (demoInput) {
    demoInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') landingAsk();
    });
  }
});

document.addEventListener('DOMContentLoaded', () => {
  // Core setup
  initMouseTracking();
  initHamburger();
  initButtonEffects();
  initSmoothScroll();

  // Visuals
  createLeaves();
  initThreeHero();
  initNeuralCanvas();
  initEcoCanvas();

  // Scroll-driven
  initScrollReveal();
  initRootTransitions();
  animateCounters();

  // Backend health check (run immediately + every 30s)
  checkBackendHealth();
  setInterval(checkBackendHealth, 30000);

  // Event listeners
  window.addEventListener('scroll', () => {
    handleNavbarScroll();
    handleHeroScroll();
    updateScrollProgress();
  }, { passive: true });

  window.addEventListener('resize', () => {
    createLeaves();
    const viz = document.getElementById('ecosystemViz');
    if (ecoCanvas && viz) {
      ecoCanvas.width  = viz.clientWidth;
      ecoCanvas.height = viz.clientHeight;
    }
  });

  // Start main render loop
  requestAnimationFrame(mainLoop);

  // Trigger initial states
  handleNavbarScroll();
  handleHeroScroll();
  updateScrollProgress();
});
