# IP-SAKTI Sahayak — System Architecture (MVP)

Multilingual, RAG-based AI assistant for IP and regulatory guidance in Ayurveda.
This document covers the **MVP architecture only**, broken down by feature.

---

## 1. Core RAG Pipeline (MVP backbone)

```mermaid
flowchart TD
    U[User Query] --> LI[Language Detection / Input Normalization]
    LI --> QC[Query Classifier]
    QC --> RET[Retriever - Vector Search]
    RET --> CORPUS[(Curated Corpus<br/>Statutes, Rules, Treaties,<br/>Pharmacopoeial Standards)]
    RET --> RERANK[Reranker / Top-k Selection]
    RERANK --> LLM[LLM - Grounded Answer Generation]
    LLM --> CITE[Citation Extractor<br/>maps claims to source spans]
    CITE --> CONF[Confidence Scorer]
    CONF --> DISC[Disclaimer Injector<br/>'information, not legal advice']
    DISC --> OUT[Response to User]
```

---

## 2. Formulation Classification Flow

```mermaid
flowchart TD
    START[User describes product] --> Q1{Drawn from<br/>First-Schedule text?}
    Q1 -->|Yes| CLASS[Classical/Generic Medicine]
    Q1 -->|No| Q2{Has proprietary<br/>formulation?}
    Q2 -->|Yes| PROP[Patent/Proprietary Medicine]
    Q2 -->|No| Q3{Requires safety/<br/>efficacy proof?}
    Q3 -->|Yes| NEWDRUG[New/Non-Classical Drug]
    Q3 -->|No| Q4{Isolated/standardized<br/>phytoconstituent?}
    Q4 -->|Yes| PHYTO[Phytopharmaceutical]
    Q4 -->|No| Q5{Food/nutraceutical<br/>or cosmetic use?}
    Q5 -->|Food| AAHAR[Ayurveda-Aahar / Nutraceutical]
    Q5 -->|Cosmetic| COSM[Cosmetic]

    CLASS --> POSTURE[Route to IP + ABS Posture Engine]
    PROP --> POSTURE
    NEWDRUG --> POSTURE
    PHYTO --> POSTURE
    AAHAR --> POSTURE
    COSM --> POSTURE
```

---

## 3. Jurisdiction Toggle (India vs International)

```mermaid
flowchart TD
    QRY[Classified Query] --> TOGGLE{Jurisdiction<br/>Toggle}
    TOGGLE -->|India| NAT[National Corpus Slice<br/>Patents Act + 2024 Rules,<br/>GI, Trade Marks, Designs,<br/>Copyright, Plant Variety,<br/>Biological Diversity Act,<br/>Drugs & Cosmetics Act,<br/>DMR Act, FSSAI Ayurveda-Aahar]
    TOGGLE -->|International| INTL[International Corpus Slice<br/>TRIPS, CBD + Nagoya,<br/>WIPO GRATK Treaty,<br/>PCT, Madrid, Hague,<br/>Budapest Treaty]
    NAT --> ANS1[National Answer Set]
    INTL --> ANS2[International Answer Set]
    ANS1 -.kept visibly separate.- ANS2
```

---

## 4. Source Citation & Trust Layer

```mermaid
flowchart TD
    GEN[Generated Answer] --> MAP[Map each claim<br/>to source document + clause]
    MAP --> VALID{Source found<br/>in corpus?}
    VALID -->|Yes| ATTACH[Attach citation:<br/>statute/rule/treaty article + record]
    VALID -->|No| ABSTAIN[Trigger Safe Abstention]
    ATTACH --> SCORE[Confidence Indicator]
    SCORE --> LOWCONF{Below threshold?}
    LOWCONF -->|Yes| ESCALATE[Escalate to Human<br/>IP Facilitator]
    LOWCONF -->|No| FINAL[Deliver Cited Response]
```

---

## 5. Corpus Ingestion (MVP scope only)

```mermaid
flowchart TD
    SRC[Official Free Sources<br/>Acts, Rules, Treaties, TKDL public data] --> INGEST[Ingestion Pipeline]
    INGEST --> CLEAN[Clean + Chunk + Version-tag]
    CLEAN --> EMBED[Embedding Generation]
    EMBED --> VDB[(Vector DB)]
    CLEAN --> META[(Metadata Store<br/>version, effective date, source URL)]
```

---

## 6. Minimal System Overview (how MVP pieces connect)

```mermaid
flowchart TD
    UI[Chat UI - text, single language MVP] --> API[Backend API]
    API --> CLASSIFY[Formulation Classifier]
    API --> JURIS[Jurisdiction Toggle]
    CLASSIFY --> RAGCORE[Core RAG Pipeline]
    JURIS --> RAGCORE
    RAGCORE --> VDB[(Vector DB)]
    RAGCORE --> CITELAYER[Citation & Confidence Layer]
    CITELAYER --> UI
    CITELAYER -->|low confidence| HUMAN[Human IP Facilitator Escalation]
```

---

*Future layers (not in MVP): relational knowledge graph, agentic multi-source orchestration, ABS-compliance helper, TKDL/prior-art pointer, paid-source connectors, multilingual delivery (Bhashini), voice interface.*
