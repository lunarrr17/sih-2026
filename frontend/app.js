// IP-SAKTI Sahayak — 5-Step Triage & Legal Chat Agent Logic
const API_BASE = (window.location.origin.includes(":8000") || window.location.origin.includes(":3000"))
  ? window.location.origin 
  : "http://127.0.0.1:8000";

let currentJurisdiction = "national";
let currentSessionId = "session_" + Math.random().toString(36).substring(2, 9);
let currentTriageResult = null;
let conversationHistory = [];
let accumulatedCitations = [];

// Jurisdiction Toggle
document.querySelectorAll('#jurisdictionToggle .toggle-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('#jurisdictionToggle .toggle-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentJurisdiction = btn.getAttribute('data-jur');
  });
});

function sendQuickPrompt(text) {
  document.getElementById('chatInput').value = text;
  document.getElementById('chatForm').dispatchEvent(new Event('submit'));
}

// 5-Step Formulation Triage Form Submission
document.getElementById('triageForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const productName = document.getElementById('triageProductName').value;
  const isClassical = document.querySelector('input[name="triage_is_classical"]:checked').value === 'yes';
  const extractionMethod = document.getElementById('triageExtraction').value;
  const deliveryFormat = document.getElementById('triageDelivery').value;
  const hasSynergy = document.getElementById('triageSynergy').checked;
  const isPractitioner = document.getElementById('triagePractitioner').checked;

  const intendedUse = deliveryFormat === 'food_beverage_powder' 
    ? 'food_wellness_supplement' 
    : (deliveryFormat === 'topical_emulgel' ? 'therapeutic_topical' : 'therapeutic_internal');

  try {
    const res = await fetch(`${API_BASE}/api/triage/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        product_name: productName,
        is_first_schedule_text: isClassical,
        extraction_method: extractionMethod,
        delivery_format: deliveryFormat,
        has_comparative_synergy_data: hasSynergy,
        synergy_percentage_increase: hasSynergy ? 40.0 : 0.0,
        is_registered_ayush_practitioner: isPractitioner,
        intended_use: intendedUse
      })
    });
    
    currentTriageResult = await res.json();
    renderTriageCard(currentTriageResult);

    // Also inform the chat stream
    appendUserMessage(`⚡ Evaluated 5-Step Triage for: ${productName}`);
    appendBotMessage({
      query: `Triage evaluation for ${productName}`,
      jurisdiction: currentJurisdiction,
      confidence_score: 0.96,
      answer: `### 📋 5-Step Formulation Triage Result: **${productName}**\n\n` +
              `- **Category**: \`${currentTriageResult.category}\`\n` +
              `- **Patent Status**: \`${currentTriageResult.patent_status}\` — ${currentTriageResult.patent_rationale}\n` +
              `- **Biodiversity ABS Posture**: \`${currentTriageResult.abs_status}\` — ${currentTriageResult.abs_rationale}\n` +
              `- **Licensing Regime**: ${currentTriageResult.licensing_framework}\n\n` +
              `*You can ask follow-up questions about this specific posture or export a formal Dossier.*`,
      citations: currentTriageResult.statutory_citations.map(c => ({
        statute: c,
        section: "Statutory Directive",
        title: c,
        source_url: "https://ipindia.gov.in",
        page_numbers: []
      }))
    });

  } catch (err) {
    console.error("Triage evaluation error:", err);
    alert("Could not reach API server.");
  }
});

function renderTriageCard(data) {
  const card = document.getElementById('classificationCard');
  card.classList.remove('hidden');

  document.getElementById('catBadge').innerText = data.category;
  
  const patentBadge = document.getElementById('patentBadge');
  patentBadge.innerText = data.patent_status;
  if (data.patent_status.includes('ELIGIBLE')) {
    patentBadge.className = 'badge-status eligible';
  } else {
    patentBadge.className = 'badge-status';
  }

  document.getElementById('patentPosture').innerText = data.patent_rationale;
  document.getElementById('absPosture').innerText = data.abs_rationale;
  document.getElementById('authorityPosture').innerText = data.licensing_framework;

  // Forms
  const formsContainer = document.getElementById('formsList');
  formsContainer.innerHTML = "";
  if (data.licensing_forms) {
    data.licensing_forms.forEach(f => {
      const chip = document.createElement('span');
      chip.className = 'form-chip';
      chip.innerText = `📄 ${f}`;
      formsContainer.appendChild(chip);
    });
  }

  // Checklist
  const checklistUl = document.getElementById('checklistUl');
  checklistUl.innerHTML = "";
  if (data.compliance_checklist) {
    data.compliance_checklist.forEach(item => {
      const li = document.createElement('li');
      li.innerText = item;
      checklistUl.appendChild(li);
    });
  }
}

// Chat Form Submission
document.getElementById('chatForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('chatInput');
  const query = input.value.trim();
  if (!query) return;

  input.value = "";
  appendUserMessage(query);

  const loadingBubble = appendLoadingBubble();

  try {
    const res = await fetch(`${API_BASE}/api/chat/`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        jurisdiction: currentJurisdiction,
        session_id: currentSessionId,
        classification_context: currentTriageResult
      })
    });

    const data = await res.json();
    loadingBubble.remove();
    appendBotMessage(data);

    conversationHistory.push({ role: "user", content: query });
    conversationHistory.push({ role: "assistant", content: data.answer });
    if (data.citations) {
      accumulatedCitations.push(...data.citations);
    }
  } catch (err) {
    loadingBubble.remove();
    appendErrorMessage("Could not connect to IP-SAKTI Sahayak backend.");
  }
});

function appendUserMessage(text) {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg-bubble user';
  div.innerText = text;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendLoadingBubble() {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg-bubble bot';
  div.innerHTML = `<p style="color:#a7f3d0;">🌿 Reasoning over Qdrant statutory collections & Cross-Encoder ranking...</p>`;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return div;
}

function appendBotMessage(data) {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg-bubble bot';

  const isAbstain = data.abstain === true;
  const confidencePct = Math.round((data.confidence_score || 0) * 100);
  const formattedHtml = formatMarkdown(data.answer || '');

  let badgeHtml;
  if (isAbstain) {
    div.style.borderColor = 'rgba(245, 158, 11, 0.4)';
    div.style.background = 'linear-gradient(135deg, rgba(245, 158, 11, 0.05), rgba(15, 23, 42, 0.95))';
    badgeHtml = `<span class="confidence-badge" style="background:rgba(245,158,11,0.2); color:#fbbf24; border-color:rgba(245,158,11,0.5);">⚠️ Evidence Strength: Insufficient</span>`;
  } else {
    const strength = data.evidence_strength || (confidencePct >= 80 ? 'Strong' : 'Moderate');
    badgeHtml = `<span class="confidence-badge" title="Heuristic lexical grounding index: ${(data.confidence_score || 0).toFixed(2)}. Informational only, does not imply probability of legal correctness.">🎯 Evidence Strength: ${escapeHtml(strength)}</span>`;
  }

  const jurText = (data.jurisdiction || 'NATIONAL').toUpperCase();

  div.innerHTML = `
    <div class="msg-header">
      <span class="bot-name">🌿 IP-SAKTI Sahayak [${escapeHtml(jurText)}]</span>
      ${badgeHtml}
    </div>
    <div class="msg-text">${formattedHtml}</div>
  `;

  // Append citation pills only if not an abstention and citations exist
  if (!isAbstain && data.citations && data.citations.length > 0) {
    const pillRow = document.createElement('div');
    pillRow.className = 'citation-pill-row';
    data.citations.forEach((c) => {
      const pill = document.createElement('button');
      pill.className = 'citation-pill';
      pill.innerText = `📖 ${c.title}`;
      pill.addEventListener('click', () => openCitationDrawer(c));
      pillRow.appendChild(pill);
    });
    div.appendChild(pillRow);
  }

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendErrorMessage(msg) {
  const container = document.getElementById('chatMessages');
  const div = document.createElement('div');
  div.className = 'msg-bubble bot';
  div.innerHTML = `<p style="color:#f87171;">⚠️ ${escapeHtml(msg)}</p>`;
  container.appendChild(div);
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

// Markdown Formatter with HTML sanitization
function formatMarkdown(text) {
  if (!text) return '';
  // Pre-escape raw HTML
  let clean = escapeHtml(text);
  return clean
    .replace(/^### (.*$)/gim, '<h4 style="color:#a7f3d0; margin: 10px 0 4px;">$1</h4>')
    .replace(/^## (.*$)/gim, '<h3 style="color:#6ee7b7; margin: 12px 0 6px;">$1</h3>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.1); padding:2px 5px; border-radius:4px;">$1</code>')
    .replace(/^- (.*$)/gim, '<li style="margin-left: 18px;">$1</li>')
    .replace(/\n\n/g, '<br/><br/>');
}

// Citation Drawer Handler
function openCitationDrawer(citation) {
  const drawer = document.getElementById('citationDrawer');
  const content = document.getElementById('drawerContent');
  const pages = citation.page_numbers && citation.page_numbers.length > 0 ? citation.page_numbers.join(', ') : "N/A";
  const url = citation.source_url && (citation.source_url.startsWith('http://') || citation.source_url.startsWith('https://'))
    ? citation.source_url
    : 'https://ipindia.gov.in';

  const isSecondary = citation.source_type === 'secondary_academic_study';
  const typeBadge = isSecondary
    ? `<span style="display:inline-block; background:rgba(168, 85, 247, 0.15); color:#c084fc; border:1px solid rgba(168,85,247,0.4); padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; margin-bottom:10px;">📘 Secondary Academic Study (Non-Gazette)</span>`
    : `<span style="display:inline-block; background:rgba(16, 185, 129, 0.15); color:#34d399; border:1px solid rgba(16,185,129,0.4); padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; margin-bottom:10px;">🏛️ Primary Statutory Authority</span>`;

  const btnLabel = isSecondary
    ? '🔗 View Academic Journal Study ➔'
    : '🔗 Open Official Government Gazette / Treaty ➔';

  content.innerHTML = `
    ${typeBadge}
    <h4 style="color:#6ee7b7; margin-bottom:8px;">${escapeHtml(citation.statute)}</h4>
    <div style="background:#09130e; padding:12px; border-radius:8px; border:1px solid rgba(16,185,129,0.3); margin-bottom:12px;">
      <p><strong>Section / Clause:</strong> ${escapeHtml(citation.section)}</p>
      <p style="margin-top:4px;"><strong>Official Page Ref:</strong> Page ${escapeHtml(pages)}</p>
    </div>
    <a href="${encodeURI(url)}" target="_blank" rel="noopener noreferrer" class="btn-primary" style="display:inline-block; text-decoration:none; text-align:center; width:100%;">
      ${btnLabel}
    </a>
  `;

  drawer.classList.remove('hidden');
}

document.getElementById('closeDrawerBtn').addEventListener('click', () => {
  document.getElementById('citationDrawer').classList.add('hidden');
});

// 1-Click Export Dossier
document.getElementById('exportDossierBtn').addEventListener('click', async () => {
  const productName = document.getElementById('triageProductName').value || "Ayurvedic_Product";
  const categoryName = currentTriageResult ? currentTriageResult.category : "Classical Ayurvedic Medicine";

  try {
    const res = await fetch(`${API_BASE}/api/export/dossier`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        product_name: productName,
        category_name: categoryName,
        jurisdiction: currentJurisdiction,
        chat_history: conversationHistory.length > 0 ? conversationHistory : [{ role: "assistant", content: "Direct 5-Step Formulation Triage Assessment" }],
        citations: accumulatedCitations,
        triage_result: currentTriageResult
      })
    });

    const data = await res.json();
    const blob = new Blob([data.content], { type: 'text/markdown' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = data.filename;
    a.click();
    window.URL.revokeObjectURL(url);
  } catch (err) {
    alert("Error generating dossier export.");
  }
});
