# IP-SAKTI Sahayak — System Architecture

**IP-SAKTI Sahayak** is a multilingual, source-grounded (RAG-based) AI assistant for Intellectual Property (IPR) and regulatory guidance in Ayurveda, operating across distinct National and International legal regimes.

---

## 1. High-Level Architectural Blueprint

```mermaid
flowchart TD
    subgraph UI ["Client & Interaction Layer"]
        UQ["User Query / Formulation Description"]
        JT["Jurisdiction Toggle: National 🇮🇳 | International 🌐"]
        WIZ["Formulation Classification Wizard"]
        LANG["Multilingual / Voice Interface - Bhashini"]
    end

    subgraph ORCHESTRATION ["Agentic & Routing Layer"]
        INP["Input Normalization & Language Transliteration"]
        QC["Query & Intent Classifier"]
        FC["Formulation Classifier Engine"]
        ABS["ABS & Biodiversity Pre-screener"]
        JT_ROUTER{Jurisdiction Router}
    end

    subgraph RETRIEVAL ["Knowledge & Retrieval Layer (Hybrid & GraphRAG)"]
        subgraph NATIONAL_SLICE ["National Corpus (India)"]
            NAT_VEC[("Vector DB - National")]
            NAT_BM25[("BM25 Sparse Index")]
            NAT_DOCS["Patents Act 1970 + 2024 Rules<br/>Biological Diversity Act 2023/2024<br/>Drugs & Cosmetics Act First Schedule<br/>DMR Act 1954<br/>FSSAI Ayurveda-Aahar Regulations<br/>GI, TM, Designs & PPV&FR Acts"]
        end

        subgraph INTL_SLICE ["International Corpus (Global / Export)"]
            INTL_VEC[("Vector DB - International")]
            INTL_BM25[("BM25 Sparse Index")]
            INTL_DOCS["WIPO GRATK Treaty 2024<br/>CBD & Nagoya Protocol<br/>TRIPS Agreement & Budapest Treaty<br/>PCT & Madrid Systems<br/>US FDA Botanical Guidance / DSHEA<br/>EU THMPD Directive 2004/24/EC"]
        end

        KG[("Relational Legal Knowledge Graph<br/>Herbs &harr; Formulations &harr; Sections &harr; Forms")]
        RERANK["Cross-Encoder Reranker & Context Assembler"]
    end

    subgraph SYNTHESIS ["Grounded Synthesis & Trust Layer"]
        LLM["Grounded LLM Generator"]
        CITE["Span-Level Citation Validator & Linker"]
        CONF["Confidence Scorer"]
        DISC["Standing Disclaimer Injector:<br/>'Information, not legal advice'"]
        GUARD{Confidence Check & Safe Abstention}
        ABSTAIN["Safe Abstention Handler"]
        ESCALATE["Escalation to Human IP Facilitator Dossier"]
    end

    UQ --> INP
    WIZ --> FC
    JT --> JT_ROUTER
    LANG --> INP

    INP --> QC
    QC --> JT_ROUTER
    FC -.Classification Context.-> JT_ROUTER
    ABS -.ABS Context.-> JT_ROUTER

    JT_ROUTER -->|National / India| NATIONAL_SLICE
    JT_ROUTER -->|International| INTL_SLICE

    NATIONAL_SLICE --> RERANK
    INTL_SLICE --> RERANK
    KG -.Multi-hop Traversal.-> RERANK

    RERANK --> LLM
    LLM --> CITE
    CITE --> CONF
    CONF --> GUARD

    GUARD -->|Pass / Grounded| DISC
    GUARD -->|Fail / Out-of-Scope| ABSTAIN
    GUARD -->|Low Confidence / User Trigger| ESCALATE

    DISC --> UI
    ABSTAIN --> UI
    ESCALATE --> UI
```

---

## 2. Core Subsystems & Components

### 1. Formulation Classification & IP/ABS Posture Engine
Because IPR for Ayurvedic products is intrinsically tied to drug classification, the system executes a 5-step triage decision tree:

```mermaid
flowchart TD
    START[User describes product] --> Q1{Drawn verbatim from<br/>First-Schedule text?}
    Q1 -->|Yes| CLASS[1. Classical / Generic Medicine]
    Q1 -->|No| Q2{Has proprietary<br/>formulation / ratio / vehicle?}
    Q2 -->|Yes| PROP[2. Patent or Proprietary Medicine]
    Q2 -->|No| Q3{Requires clinical proof<br/>of safety & efficacy?}
    Q3 -->|Yes| NEWDRUG[3. New / Non-Classical Drug]
    Q3 -->|No| Q4{Isolated / standardized<br/>phytoconstituent fraction?}
    Q4 -->|Yes| PHYTO[4. Phytopharmaceutical Drug]
    Q4 -->|No| Q5{Intended Use?}
    Q5 -->|Food / Nutrition| AAHAR[5. Ayurveda-Aahar / Nutraceutical]
    Q5 -->|Beautification / Topical| COSM[6. Ayurvedic Cosmetic]

    CLASS --> POSTURE[IP & ABS Posture Engine]
    PROP --> POSTURE
    NEWDRUG --> POSTURE
    PHYTO --> POSTURE
    AAHAR --> POSTURE
    COSM --> POSTURE
```

#### Category Posture Matrix:
| Category | Governing Regulatory Act | Licensing Authority | Patent Potential & Key Hurdles | ABS & Biodiversity Posture |
|---|---|---|---|---|
| **Classical Medicine** | Drugs & Cosmetics Act 1940 (Chapter IV-A) | State Licensing Authority (SLA) | **Barred under Section 3(p)** as Traditional Knowledge (defended via TKDL). Eligible for Trademarks (Class 5) & GI. | **Exempted** for registered AYUSH practitioners under 2023 BD Amendment Act. Commercial mass-production requires SBB intimation. |
| **Patent or Proprietary (P&P)** | D&C Act 1940 (Rule 158B) | State Licensing Authority (SLA) | Conditional. Must prove **unexpected synergy under Section 3(e)** and therapeutic enhancement under **Section 3(d)**. | SBB prior intimation for Indian entities; NBA Form I approval for foreign entities. |
| **New Drug** | New Drugs and Clinical Trials Rules 2019 | DCGI / CDSCO | High patent potential. Requires Phase I–III clinical trials. | Mandatory NBA Form III clearance prior to patent grant. |
| **Phytopharmaceutical** | New Drugs Rules 2019 / Schedule Y | CDSCO | Strong patentability for novel extraction process, standardized fraction ($\ge 4$ biomarkers). | Mandatory NBA Form I / III approvals and benefit sharing. |
| **Ayurveda-Aahar** | FSSAI Ayurveda Aahar Reg. 2022 | FSSAI | Low patentability. IP centered on Trademarks (Class 29/30) and Trade Dress. **No medicinal cure claims allowed**. | SBB intimation for commercial sourcing of biological food ingredients. |
| **Cosmetic** | Cosmetics Rules 2020 / D&C Act | SLA (Cosmetics Wing) | IP centered on Trademarks (Class 3), Packaging Designs (Designs Act 2000), and Trade Dress. | SBB intimation for commercial sourcing. |

---

## 3. Dual-Jurisdiction Architecture

The system maintains strict segregation between National and International legal corpora so advice is never conflated:

```mermaid
flowchart TD
    QUERY[Incoming User Query] --> TOGGLE{Jurisdiction Switch}
    
    TOGGLE -->|India / National| NAT_PIPE[National Retrieval Pipeline]
    TOGGLE -->|International| INTL_PIPE[International Retrieval Pipeline]
    TOGGLE -->|Comparative / Both| DUAL_PIPE[Parallel Dual Pipeline]
    
    subgraph NAT_PIPE [National Regimes]
        N1[Patents Act 1970 & 2024 Rules]
        N2[Biological Diversity Act 2002 / 2023 Amendment / 2024 Rules]
        N3[Drugs & Cosmetics Act 1940 & First Schedule Texts]
        N4[Drugs & Magic Remedies Act 1954 & ASCI/CCPA Guidelines]
        N5[FSSAI Ayurveda-Aahar Regulations 2022]
        N6[GI Act 1999, TM Act 1999, Designs Act 2000, PPV&FR Act 2001]
    end
    
    subgraph INTL_PIPE [International Regimes]
        I1[WIPO GRATK Treaty 2024 - Mandatory Origin Disclosure]
        I2[CBD & Nagoya Protocol on Access and Benefit Sharing]
        I3[TRIPS Agreement Art 27.3b & Budapest Treaty]
        I4[PCT & Madrid Systems for International Filings]
        I5[US FDA Botanical Guidance & DSHEA 1994]
        I6[EU Traditional Herbal Medicinal Products Directive 2004/24/EC]
    end
    
    NAT_PIPE --> OUT_NAT[National Grounded Answer]
    INTL_PIPE --> OUT_INTL[International Grounded Answer]
    DUAL_PIPE --> OUT_DUAL[Visibly Segregated Dual-Column Comparative Output]
```

---

## 4. Trust, Source Citation & Guardrails Layer

```mermaid
flowchart TD
    GEN[Generated Response] --> EXTRACT[Extract Claim-to-Source Spans]
    EXTRACT --> VERIFY{Verified in Curated Corpus?}
    
    VERIFY -->|No / Unsupported| ABSTAIN[Trigger Safe Abstention]
    VERIFY -->|Yes| ATTACH[Attach Exact Citations:<br/>Act / Section / Rule / Article + Gazette URL]
    
    ATTACH --> SCORE[Compute Grounding Confidence Score]
    SCORE --> CONF_CHECK{Score >= Threshold?}
    
    CONF_CHECK -->|No| ESCALATE[Escalate to Human IP Facilitator Brief]
    CONF_CHECK -->|Yes| DISC[Inject Standing Legal Disclaimer:<br/>'Information, not legal advice']
    
    DISC --> DELIVER[Deliver Verified Response to User]
```

---

## 5. Phased Evolution Roadmap

```
Phase 1: Architecture & Minimum Working MVP (Current)
├── Clean Scaffolding & Virtual Environment
├── Curated Statutory Corpus (National & International Markdown datasets with 2024 rules)
├── Section-aware Corpus Ingestion & Hybrid BM25/Vector Sliced Retriever
├── 5-Step Formulation Classifier & ABS Engine
├── Grounded RAG Orchestrator with Citation Extractor & Guardrails
├── REST API Layer & Responsive Frontend with Jurisdiction Toggle & Citation Drawer
└── Automated Contract & Flow Test Suite

Phase 2: Relational Legal Knowledge Graph (GraphRAG)
├── Entities: Medicinal Plants, Classical Formulations, Bio-markers, Statutes, Sections, Forms
├── Edges: REQUIRES_ABS, BARS_PATENT_SEC_3P, OVERCOMES_ADMIXTURE, GOVERNED_BY
└── Multi-hop statutory reasoning across acts

Phase 3: Multilingual (Bhashini) & Voice Integration
├── Integration with Bhashini API (IndicTrans2, IndicASR, IndicTTS)
├── Sanskrit/Ayurvedic technical terminology preservation
└── Voice querying for AYUSH MSMEs, practitioners, and cultivators

Phase 4: Official Registry Connectors & Governance
├── Free database pointers (InPASS, Patentscope, TKDL public records, NBA e-filing)
├── 1-Click "IP Facilitator Handoff Dossier" Export (PDF/JSON)
└── DPDP Act 2023 compliance, audit logging & session isolation
```
