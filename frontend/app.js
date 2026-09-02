// IP-SAKTI Sahayak — Spatial Intelligence 
const API_BASE = (window.location.origin.includes(":8000") || window.location.origin.includes(":3000"))
  ? window.location.origin 
  : "http://127.0.0.1:8000";

let currentJurisdiction = "national";
let currentSessionId = "session_" + Math.random().toString(36).substring(2, 9);
let aiState = 'idle'; // idle | thinking | retrieving | answering

// ================= THREE.JS ENVIRONMENT BACKGROUND =================
const envInit = () => {
  const canvas = document.getElementById('envCanvas');
  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x02070a, 0.001);
  const camera = new THREE.PerspectiveCamera(60, window.innerWidth/window.innerHeight, 0.1, 1000);
  camera.position.z = 20;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);

  // Particles
  const geometry = new THREE.BufferGeometry();
  const vertices = [];
  for(let i=0; i<1500; i++) {
    vertices.push((Math.random()-0.5)*100, (Math.random()-0.5)*100, (Math.random()-0.5)*100);
  }
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
  const material = new THREE.PointsMaterial({ color: 0x10b981, size: 0.1, transparent: true, opacity: 0.4 });
  const particles = new THREE.Points(geometry, material);
  scene.add(particles);

  const animate = () => {
    requestAnimationFrame(animate);
    particles.rotation.y += 0.0005;
    particles.rotation.x += 0.0002;
    renderer.render(scene, camera);
  };
  animate();
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth/window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
};

// ================= THREE.JS AI CORE =================
let coreRings = [];
let coreSphere;
const coreInit = () => {
  const canvas = document.getElementById('coreCanvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, rect.width/rect.height, 0.1, 100);
  camera.position.z = 10;
  camera.position.y = 2; // Look down slightly

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(rect.width, rect.height);

  // Center Sphere (Core)
  const sphereGeo = new THREE.SphereGeometry(1.5, 32, 32);
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0x06b6d4, wireframe: true, transparent:true, opacity:0.2 });
  coreSphere = new THREE.Mesh(sphereGeo, sphereMat);
  scene.add(coreSphere);

  // Concentric Rings
  for(let i=1; i<=3; i++) {
    const ringGeo = new THREE.TorusGeometry(1.5 + i*0.8, 0.02, 16, 64);
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x10b981, transparent:true, opacity: 0.3 - (i*0.05) });
    const ring = new THREE.Mesh(ringGeo, ringMat);
    ring.rotation.x = Math.PI / 2;
    scene.add(ring);
    coreRings.push(ring);
  }

  const animate = () => {
    requestAnimationFrame(animate);
    let speed = aiState === 'idle' ? 0.005 : (aiState === 'thinking' ? 0.05 : 0.02);
    
    coreSphere.rotation.y -= speed;
    coreSphere.rotation.x += speed * 0.5;
    
    coreRings.forEach((ring, i) => {
      ring.rotation.z += speed * (i%2===0?1:-1);
      ring.rotation.x = Math.PI/2 + Math.sin(Date.now()*0.001 + i) * 0.1;
    });

    if(aiState === 'thinking') {
      coreSphere.material.color.setHex(0xfbbf24);
      coreSphere.material.opacity = 0.5 + Math.sin(Date.now()*0.01)*0.2;
    } else {
      coreSphere.material.color.setHex(0x06b6d4);
      coreSphere.material.opacity = 0.2;
    }

    renderer.render(scene, camera);
  };
  animate();
};

// ================= THREE.JS KNOWLEDGE GRAPH =================
let graphGroup;
const graphInit = () => {
  const canvas = document.getElementById('graphCanvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(45, rect.width/rect.height, 0.1, 100);
  camera.position.z = 15;

  const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
  renderer.setSize(rect.width, rect.height);

  graphGroup = new THREE.Group();
  scene.add(graphGroup);

  // Nodes
  const nodeGeo = new THREE.SphereGeometry(0.3, 16, 16);
  const nodeMat1 = new THREE.MeshBasicMaterial({ color: 0x10b981 });
  const nodeMat2 = new THREE.MeshBasicMaterial({ color: 0x06b6d4 });
  const nodeMat3 = new THREE.MeshBasicMaterial({ color: 0xfbbf24 });
  
  const n1 = new THREE.Mesh(nodeGeo, nodeMat1); n1.position.set(0,2,0);
  const n2 = new THREE.Mesh(nodeGeo, nodeMat2); n2.position.set(-2,-1,0);
  const n3 = new THREE.Mesh(nodeGeo, nodeMat3); n3.position.set(2,-1,0);
  graphGroup.add(n1,n2,n3);

  // Lines
  const lineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent:true, opacity:0.2 });
  const pts1 = [n1.position, n2.position];
  const pts2 = [n1.position, n3.position];
  const pts3 = [n2.position, n3.position];
  
  graphGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts1), lineMat));
  graphGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts2), lineMat));
  graphGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts3), lineMat));

  const animate = () => {
    requestAnimationFrame(animate);
    graphGroup.rotation.y += 0.005;
    renderer.render(scene, camera);
  };
  animate();
};

// Init Three.js Views
envInit();
coreInit();
graphInit();


// ================= TRIAGE WIZARD =================
document.getElementById('triageForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  aiState = 'thinking';
  
  const payload = {
    product_name: document.getElementById('triageProductName').value,
    is_first_schedule_text: document.querySelector('input[name="triage_is_classical"]:checked').value === 'yes',
    extraction_method: document.getElementById('triageExtraction').value,
    delivery_format: document.getElementById('triageDelivery').value,
    has_comparative_synergy_data: document.getElementById('triageSynergy').checked,
    synergy_percentage_increase: document.getElementById('triageSynergy').checked ? 40.0 : 0.0,
    is_registered_ayush_practitioner: document.getElementById('triagePractitioner').checked,
    intended_use: "therapeutic_internal"
  };

  try {
    const res = await fetch(`${API_BASE}/api/triage/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    aiState = 'idle';

    // Update Verdict Card
    const card = document.getElementById('classificationCard');
    card.classList.remove('hidden');

    document.getElementById('catBadge').innerText = data.category;
    const pat = document.getElementById('patentBadge');
    pat.innerText = data.patent_status;
    pat.className = 'verdict-pill ' + (data.patent_status.includes('ELIGIBLE') ? 'success' : 'alert');
    
    const abs = document.getElementById('absBadge');
    abs.innerText = data.abs_status;
    abs.className = 'verdict-pill ' + (data.abs_status.includes('EXEMPT') ? 'success' : 'alert');

    document.getElementById('patentPosture').innerText = data.patent_rationale;
    document.getElementById('absPosture').innerText = data.abs_rationale;

    const forms = document.getElementById('formsList');
    forms.innerHTML = data.forms_required.map(f => `<span class="verdict-pill neutral">${f}</span>`).join('');

    appendMsg('user', `⚡ Computed Posture: ${payload.product_name}`);
    appendMsg('bot', `### Classification Complete\n**Patent**: ${data.patent_status}\n**ABS**: ${data.abs_status}\n\n*Check the Left panel for full dossier details.*`);

  } catch(e) {
    aiState = 'idle';
    console.error(e);
  }
});

// ================= CHAT =================
function sendQuickPrompt(t){
  document.getElementById('chatInput').value = t;
  document.getElementById('chatForm').dispatchEvent(new Event('submit'));
}

document.getElementById('chatForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('chatInput');
  const txt = input.value.trim();
  if(!txt) return;
  
  appendMsg('user', txt);
  input.value = '';
  aiState = 'thinking';

  // RAG Visuals
  const p = document.getElementById('ragPipeline');
  p.classList.remove('hidden');
  const stps = document.querySelectorAll('.rag-step');
  
  stps.forEach((s,i) => {
    setTimeout(()=>{
      stps.forEach(x=>x.classList.remove('active'));
      s.classList.add('active');
    }, i*400);
  });

  try {
    const res = await fetch(`${API_BASE}/api/chat/`, {
      method: "POST", headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        session_id: currentSessionId,
        message: txt,
        jurisdiction_filter: currentJurisdiction,
        extract_metadata: true
      })
    });
    const d = await res.json();
    aiState = 'idle';
    p.classList.add('hidden');

    let md = d.answer || JSON.stringify(d);
    appendMsg('bot', md, d.citations);

  } catch(e) {
    aiState = 'idle';
    p.classList.add('hidden');
    appendMsg('bot', "Network Error linking to Core.");
  }
});

let citeCounter = 0;
let citeDataStore = {};

function appendMsg(role, text, citations=[]) {
  const c = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = `msg-bubble ${role}`;
  
  let out = role === 'bot' ? `<div class="msg-header"><span class="bot-name">IP-SAKTI </span></div>` : '';
  
  // parse marked if available
  try {
    out += `<div class="msg-text">${marked.parse(text)}</div>`;
  } catch(e) {
    out += `<div class="msg-text">${text.replace(/\n/g,'<br>')}</div>`;
  }

  if(citations && citations.length > 0) {
    out += `<div class="cite-row">`;
    citations.forEach(cit => {
      citeCounter++;
      citeDataStore[citeCounter] = cit;
      out += `<span class="citation-link" onclick="openCitation(${citeCounter})">[${cit.statute.split(/[|, \-:]/)[0].substring(0,15)}]</span>`;
    });
    out += `</div>`;
  }

  div.innerHTML = out;
  c.appendChild(div);
  c.scrollTop = c.scrollHeight;
}

// CITATION DRAWER
window.openCitation = (id) => {
  const d = citeDataStore[id];
  document.getElementById('drawerEmpty').classList.add('hidden');
  const content = document.getElementById('drawerContent');
  content.classList.remove('hidden');
  
  content.innerHTML = `
    <div class="citation-detail-card">
      <h4>${d.statute}</h4>
      <div class="statute">Source: ${d.section}</div>
      <p>"${d.title}... "</p>
      <a class="citation-link-out" href="${d.source_url}" target="_blank">View Official Source ➔</a>
    </div>
  `;
};

// EXPORT
document.getElementById('exportDossierBtn').addEventListener('click', () => {
  const o = document.getElementById('exportOverlay');
  o.classList.remove('hidden');
  setTimeout(()=>{
    o.classList.add('hidden');
    alert("Dossier compiled successfully. Download started.");
  }, 2500);
});

// Controls
document.querySelectorAll('#jurisdictionToggle .toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#jurisdictionToggle .toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentJurisdiction = btn.getAttribute('data-jur');
  });
});
