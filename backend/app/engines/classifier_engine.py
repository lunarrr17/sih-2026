import re
import unicodedata
from typing import Dict, Any, List, Optional, Tuple, Set

from backend.app.rag.schemas import (
    SubjectType,
    SubstanceOrigin,
    ProcessType,
    TraditionalKnowledgeSignal,
    UserIntent,
    ConfidenceTier,
    RoutingHints,
    FormulationIntelligence,
)

# Legacy schemas retained solely for backward API signature compatibility.
# NOTE: Active Phase 4 pipeline uses FormulationIntelligence exclusively.
from pydantic import BaseModel, Field

class FormulationInput(BaseModel):
    """DEPRECATED: Legacy input model retained only for backward signature compatibility."""
    product_name: str = "Chyawanprash Awaleha"
    is_first_schedule: bool = True
    has_proprietary_formulation: bool = False
    requires_clinical_safety_proof: bool = False
    is_isolated_phytoconstituent: bool = False
    product_use_category: str = "therapeutic"

class ClassificationResult(BaseModel):
    """DEPRECATED: Legacy result model retained only for backward signature compatibility."""
    category_id: str
    category_name: str
    statutory_definition: str
    governing_regime: str
    licensing_authority: str
    ip_posture: Dict[str, str]
    abs_posture: Dict[str, str]
    statutory_citations: List[str]


class FormulationClassifierEngine:
    """
    Deterministic, explainable, and testable Formulation Intelligence and Classification Engine.
    Extracts structured entities, detects substance origins, process types, novelty cues,
    and traditional knowledge markers from natural language queries.

    ARCHITECTURAL PRINCIPLE:
    Classification is an input to retrieval and routing; it NEVER establishes legal conclusions.
    """

    CLASSICAL_FORMULATIONS = {
        "chyawanprash": {
            "canonical": "Chyawanprash",
            "aliases": ["chyawanprash", "chyavanprash", "chyavanaprasha", "cyavanaprasa", "chyavanaprasham", "chyawanprash awaleha"],
            "texts": ["Charaka Samhita", "First Schedule"],
            "dosage_form": "avaleha"
        },
        "triphala": {
            "canonical": "Triphala",
            "aliases": ["triphala", "trifala", "triphala churna", "triphala choorna", "triphala guggulu"],
            "texts": ["Charaka Samhita", "Sushruta Samhita", "AFI"],
            "dosage_form": "churna"
        },
        "trikatu": {
            "canonical": "Trikatu",
            "aliases": ["trikatu", "trikatu churna", "trikatu choorna"],
            "texts": ["Charaka Samhita", "AFI"],
            "dosage_form": "churna"
        },
        "sitopaladi": {
            "canonical": "Sitopaladi Churna",
            "aliases": ["sitopaladi", "sitopaladi churna", "sitopaladi choorna"],
            "texts": ["Sharangadhara Samhita", "AFI"],
            "dosage_form": "churna"
        },
        "brahmi ghrita": {
            "canonical": "Brahmi Ghrita",
            "aliases": ["brahmi ghrita", "brahmi ghee", "brahmighritam", "brahmi taila"],
            "texts": ["Ashtanga Hridaya", "AFI"],
            "dosage_form": "ghrita"
        },
        "ashwagandharishta": {
            "canonical": "Ashwagandharishta",
            "aliases": ["ashwagandharishta", "asvagandharista", "ashwagandha arishta", "aswagandharishtam"],
            "texts": ["Bhaishajya Ratnavali", "AFI"],
            "dosage_form": "arishta"
        },
        "dashmoolarishta": {
            "canonical": "Dashmoolarishta",
            "aliases": ["dashmoolarishta", "dasamoolarishtam", "dashmula arishta", "dashamoolarishta"],
            "texts": ["Sharangadhara Samhita", "AFI"],
            "dosage_form": "arishta"
        },
        "saraswatarishta": {
            "canonical": "Saraswatarishta",
            "aliases": ["saraswatarishta", "saraswatarishtam", "saraswat arishta"],
            "texts": ["Bhaishajya Ratnavali", "AFI"],
            "dosage_form": "arishta"
        },
        "avipattikar": {
            "canonical": "Avipattikar Churna",
            "aliases": ["avipattikar", "avipattikar churna", "avipattikara"],
            "texts": ["Bhaishajya Ratnavali", "AFI"],
            "dosage_form": "churna"
        },
        "chandraprabha": {
            "canonical": "Chandraprabha Vati",
            "aliases": ["chandraprabha", "chandraprabha vati", "chandraprabhavati"],
            "texts": ["Sharangadhara Samhita", "AFI"],
            "dosage_form": "vati"
        },
        "yograj guggulu": {
            "canonical": "Yograj Guggulu",
            "aliases": ["yograj guggulu", "mahayograj guggulu", "yogaraja guggulu"],
            "texts": ["Bhaishajya Ratnavali", "AFI"],
            "dosage_form": "vati"
        },
        "kaishore guggulu": {
            "canonical": "Kaishore Guggulu",
            "aliases": ["kaishore guggulu", "kaisore guggulu", "kaisoraguggulu"],
            "texts": ["Bhaishajya Ratnavali", "AFI"],
            "dosage_form": "vati"
        },
        "haridra khanda": {
            "canonical": "Haridra Khanda",
            "aliases": ["haridra khanda", "haridrakhandam", "haridra khand"],
            "texts": ["Bhaishajya Ratnavali", "AFI"],
            "dosage_form": "avaleha"
        }
    }

    KNOWN_INGREDIENTS = {
        "ashwagandha": {
            "canonical": "Ashwagandha",
            "aliases": ["ashwagandha", "asgandh", "withania somnifera", "withania", "indian ginseng"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Withania somnifera"
        },
        "turmeric": {
            "canonical": "Turmeric / Curcumin",
            "aliases": ["turmeric", "curcumin", "haridra", "haldi", "curcuma longa", "curcuminoids"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Curcuma longa"
        },
        "guduchi": {
            "canonical": "Guduchi / Giloy",
            "aliases": ["guduchi", "giloy", "amrita", "tinospora cordifolia", "tinospora"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Tinospora cordifolia"
        },
        "tulsi": {
            "canonical": "Tulsi / Holy Basil",
            "aliases": ["tulsi", "holy basil", "ocimum sanctum", "ocimum tenuiflorum"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Ocimum sanctum"
        },
        "neem": {
            "canonical": "Neem",
            "aliases": ["neem", "nimba", "azadirachta indica"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Azadirachta indica"
        },
        "amla": {
            "canonical": "Amla / Amalaki",
            "aliases": ["amla", "amalaki", "indian gooseberry", "phyllanthus emblica", "emblica officinalis"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Phyllanthus emblica (ambiguous common name)"
        },
        "haritaki": {
            "canonical": "Haritaki",
            "aliases": ["haritaki", "harad", "terminalia chebula"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Terminalia chebula"
        },
        "bibhitaki": {
            "canonical": "Bibhitaki",
            "aliases": ["bibhitaki", "baheda", "terminalia bellirica"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Terminalia bellirica"
        },
        "brahmi": {
            "canonical": "Brahmi",
            "aliases": ["brahmi", "bacopa monnieri", "centella asiatica", "mandukaparni"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Bacopa monnieri / Centella asiatica"
        },
        "guggulu": {
            "canonical": "Guggulu",
            "aliases": ["guggulu", "guggul", "commiphora mukul", "commiphora wightii"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Commiphora mukul"
        },
        "shatavari": {
            "canonical": "Shatavari",
            "aliases": ["shatavari", "satavar", "asparagus racemosus"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Asparagus racemosus"
        },
        "ginger": {
            "canonical": "Ginger / Shunthi",
            "aliases": ["ginger", "shunthi", "sonth", "ardraka", "zingiber officinale"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Zingiber officinale"
        },
        "black pepper": {
            "canonical": "Black Pepper / Maricha",
            "aliases": ["black pepper", "maricha", "piper nigrum", "kali mirch"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Piper nigrum"
        },
        "pippali": {
            "canonical": "Pippali",
            "aliases": ["pippali", "long pepper", "piper longum"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Piper longum"
        },
        "cardamom": {
            "canonical": "Cardamom / Elaichi",
            "aliases": ["cardamom", "elaichi", "ela", "elettaria cardamomum"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Elettaria cardamomum"
        },
        "cumin": {
            "canonical": "Cumin / Jeera",
            "aliases": ["cumin", "jeera", "jiraka", "cuminum cyminum"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Cuminum cyminum"
        },
        "saffron": {
            "canonical": "Saffron / Kesar",
            "aliases": ["saffron", "kesar", "kumkuma", "crocus sativus"],
            "origin": SubstanceOrigin.BOTANICAL,
            "botanical": "Crocus sativus"
        },
        "honey": {
            "canonical": "Honey / Madhu",
            "aliases": ["honey", "madhu", "makshika"],
            "origin": SubstanceOrigin.ANIMAL_DERIVED,
            "botanical": None
        },
        "ghee": {
            "canonical": "Cow Ghee / Ghrita",
            "aliases": ["ghee", "ghrita", "cow ghee", "go-ghrita"],
            "origin": SubstanceOrigin.ANIMAL_DERIVED,
            "botanical": None
        },
        "milk": {
            "canonical": "Milk / Kshira",
            "aliases": ["milk", "kshira", "cow milk"],
            "origin": SubstanceOrigin.ANIMAL_DERIVED,
            "botanical": None
        },
        "pravala": {
            "canonical": "Pravala / Coral",
            "aliases": ["pravala", "praval", "coral", "praval bhasma"],
            "origin": SubstanceOrigin.ANIMAL_DERIVED,
            "botanical": None
        },
        "mukta": {
            "canonical": "Mukta / Pearl",
            "aliases": ["mukta", "pearl", "moti", "mukta bhasma"],
            "origin": SubstanceOrigin.ANIMAL_DERIVED,
            "botanical": None
        },
        "swarna": {
            "canonical": "Swarna / Gold",
            "aliases": ["swarna", "suvarna", "gold", "swarna bhasma"],
            "origin": SubstanceOrigin.MINERAL_HERBO_MINERAL,
            "botanical": None
        },
        "rajata": {
            "canonical": "Rajata / Silver",
            "aliases": ["rajata", "silver", "rajat bhasma", "chandi"],
            "origin": SubstanceOrigin.MINERAL_HERBO_MINERAL,
            "botanical": None
        },
        "lauha": {
            "canonical": "Lauha / Iron",
            "aliases": ["lauha", "loha", "iron", "loha bhasma"],
            "origin": SubstanceOrigin.MINERAL_HERBO_MINERAL,
            "botanical": None
        },
        "abhraka": {
            "canonical": "Abhraka / Mica",
            "aliases": ["abhraka", "mica", "abhrak bhasma"],
            "origin": SubstanceOrigin.MINERAL_HERBO_MINERAL,
            "botanical": None
        },
        "parada": {
            "canonical": "Parada / Mercury",
            "aliases": ["parada", "mercury", "rasasindoor", "rasa sindoor"],
            "origin": SubstanceOrigin.MINERAL_HERBO_MINERAL,
            "botanical": None
        },
        "gandhaka": {
            "canonical": "Gandhaka / Sulphur",
            "aliases": ["gandhaka", "sulphur", "sulfur"],
            "origin": SubstanceOrigin.MINERAL_HERBO_MINERAL,
            "botanical": None
        }
    }

    DOSAGE_FORMS = {
        "churna": ["churna", "churnam", "choorna", "powder"],
        "kwath": ["kwath", "kwatha", "kashayam", "kashaya", "decoction"],
        "avaleha": ["avaleha", "avaleham", "lehya", "confection", "paste"],
        "ghrita": ["ghrita", "ghritam", "medicated ghee"],
        "taila": ["taila", "tailam", "thailam", "medicated oil", "oil"],
        "arishta": ["arishta", "arishtam", "fermented liquid"],
        "asava": ["asava", "asavam", "fermented infusion"],
        "vati": ["vati", "gutika", "tablet", "tablets", "pill", "pills"],
        "bhasma": ["bhasma", "bhasmam", "calx", "calcined ash"],
        "syrup": ["syrup", "liquid oral"],
        "capsule": ["capsule", "capsules", "hard gelatin capsule"],
        "topical": ["emulgel", "ointment", "cream", "lepa", "liniment", "topical"]
    }

    CLASSICAL_TEXTS = [
        "charaka samhita", "charaka", "sushruta samhita", "sushruta",
        "ashtanga hridaya", "ashtanga sangraha", "bhavaprakasha",
        "sharangadhara", "bhaishajya ratnavali", "sahasrayogam",
        "rasa tarangini", "rasa ratna samucchaya", "first schedule",
        "ayurvedic formulary of india", "afi", "ayurvedic pharmacopoeia of india", "api"
    ]

    NOVELTY_PROCESS_CUES = [
        "new process", "novel process", "improved extraction", "modified process",
        "new method", "improved yield", "higher purity", "novel extraction technique",
        "invented process", "new synthesis", "novel synthesis", "synthetic method",
        "novel technique", "new extraction", "new way of producing"
    ]

    OUT_OF_DOMAIN_PATTERNS = [
        r'\b(crypto\w*|bitcoin|ethereum|nft|token\w*|blockchain|web3|defi)\b',
        r'\b(tax\w*|taxation|gst|tariff\w*|customs duty|money laundering)\b',
        r'\b(weapon\w*|explosive\w*|firearm\w*|ammunition|missile\w*|bomb\w*)\b',
        r'\b(casino|gambling|betting|sports betting|lottery)\b',
        r'\b(hack\w*|crack\w*|ddos|sql injection|exploit|phishing)\b',
        r'\b(stock market prediction|crypto trading signal|forex trading)\b'
    ]

    @classmethod
    def normalize_text(cls, text: str) -> str:
        """Normalizes unicode accents, lowercases, strips excess whitespace."""
        if not text:
            return ""
        norm = unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("utf-8")
        norm = norm.lower()
        norm = re.sub(r'[\r\n\t]+', ' ', norm)
        norm = re.sub(r'[^a-z0-9\s\(\)\-\,\.\:\;]', ' ', norm)
        norm = re.sub(r'\s+', ' ', norm).strip()
        return norm

    @classmethod
    def classify_query(
        cls,
        query: str,
        jurisdiction: str = "national"
    ) -> FormulationIntelligence:
        """
        Executes deterministic formulation & intent classification.
        Produces structured metadata and routing hints without making legal conclusions.
        """
        raw_query = query.strip()
        norm_query = cls.normalize_text(raw_query)

        reasons: List[str] = []
        ambiguities: List[str] = []
        missing_info: List[str] = []
        intents: List[UserIntent] = []
        routing = RoutingHints()

        # 1. Out-of-Domain Filter
        for ood_pat in cls.OUT_OF_DOMAIN_PATTERNS:
            if re.search(ood_pat, norm_query):
                reasons.append("Query matches non-AYUSH out-of-domain pattern.")
                return FormulationIntelligence(
                    query_text=raw_query,
                    normalized_text=norm_query,
                    subject_type=SubjectType.UNKNOWN,
                    subject_confidence=ConfidenceTier.UNKNOWN,
                    user_intents=[UserIntent.UNKNOWN],
                    overall_confidence=ConfidenceTier.UNKNOWN,
                    confidence_score=0.0,
                    classification_reasons=reasons,
                    ambiguities=["Query text does not pertain to Ayurvedic IP or regulatory domains."],
                    missing_information=["Ayurvedic formulation or legal context missing."]
                )

        # 2. Entity Extraction: Formulation Names
        detected_formulation: Optional[str] = None
        alt_names: List[str] = []
        is_classical_formulation_name = False
        inherited_default_dosage_form: Optional[str] = None
        dosage_form: Optional[str] = None
        explicit_dosage_form_in_query: bool = False

        for form_key, form_data in cls.CLASSICAL_FORMULATIONS.items():
            for alias in form_data["aliases"]:
                alias_pat = rf'\b{re.escape(alias)}\b'
                if re.search(alias_pat, norm_query):
                    detected_formulation = form_data["canonical"]
                    alt_names = [a for a in form_data["aliases"] if a != alias]
                    is_classical_formulation_name = True
                    inherited_default_dosage_form = form_data.get("dosage_form")
                    reasons.append(f"Query references named formulation entity '{detected_formulation}' listed in classical treatise texts.")
                    routing.focus_terms.append(detected_formulation)
                    break
            if detected_formulation:
                break

        # Check for fuzzy/misspelled formulation names (e.g. chaywanprash, asvagandha)
        if not detected_formulation:
            misspelled_map = {
                r'\bchaywanprash\b': "Chyawanprash",
                r'\bchawanprash\b': "Chyawanprash",
                r'\btriphla\b': "Triphala",
                r'\basvagandha\b': "Ashwagandha",
                r'\bashwaganda\b': "Ashwagandha",
                r'\bgiloi\b': "Guduchi / Giloy"
            }
            for misspat, canonical_name in misspelled_map.items():
                if re.search(misspat, norm_query):
                    detected_formulation = canonical_name
                    is_classical_formulation_name = (canonical_name in ["Chyawanprash", "Triphala"])
                    if is_classical_formulation_name:
                        inherited_default_dosage_form = "avaleha" if canonical_name == "Chyawanprash" else "churna"
                    reasons.append(f"Detected likely misspelled formulation name; mapped to '{canonical_name}'.")
                    ambiguities.append(f"Query contains possible spelling variation for '{canonical_name}'.")
                    routing.focus_terms.append(canonical_name)
                    break

        # 3. Ingredients Extraction
        detected_ingredients: List[str] = []
        plant_signal = False
        animal_signal = False
        mineral_signal = False
        synthetic_signal = False

        ing_positions: List[Tuple[int, str, str, SubstanceOrigin]] = []
        for ing_key, ing_data in cls.KNOWN_INGREDIENTS.items():
            for alias in ing_data["aliases"]:
                match = re.search(rf'\b{re.escape(alias)}\b', norm_query)
                if match:
                    ing_positions.append((match.start(), ing_key, ing_data["canonical"], ing_data["origin"]))
                    break

        # Sort by appearance in query
        ing_positions.sort(key=lambda x: x[0])
        for _, ing_key, canonical, origin in ing_positions:
            if canonical not in detected_ingredients:
                detected_ingredients.append(canonical)
                routing.focus_terms.append(ing_key)
            if origin == SubstanceOrigin.BOTANICAL:
                plant_signal = True
            elif origin == SubstanceOrigin.ANIMAL_DERIVED:
                animal_signal = True
            elif origin == SubstanceOrigin.MINERAL_HERBO_MINERAL:
                mineral_signal = True

        # Check generic formulation naming if specific name wasn't detected
        if not detected_formulation:
            if detected_ingredients and any(w in norm_query for w in ["formulation", "preparation", "composition", "recipe"]):
                detected_formulation = f"{detected_ingredients[0].split('/')[0].strip()} Formulation"
                reasons.append(f"Constructed formulation identifier from identified ingredient: '{detected_formulation}'.")
            else:
                m_form = re.search(r'\b([a-z\-]+(?:\s+[a-z\-]+)?)\s+(?:formulation|preparation|composition|compound|recipe)\b', norm_query)
                if m_form:
                    cand_name = m_form.group(1).strip()
                    stop_words = ["can", "i", "patent", "a", "an", "the", "this", "that", "modified", "new", "traditional", "classical", "herbal", "polyherbal", "ayurvedic", "with", "for", "is", "does"]
                    cand_words = [w for w in cand_name.split() if w not in stop_words]
                    if cand_words:
                        detected_formulation = " ".join(cand_words).title() + " Formulation"
                        reasons.append(f"Extracted formulation reference: '{detected_formulation}'.")

        # 4. Dosage Form Extraction: Distinguish explicit query cues from inherited defaults
        explicit_dosage_form: Optional[str] = None
        for d_name, d_cues in cls.DOSAGE_FORMS.items():
            for cue in d_cues:
                if re.search(rf'\b{re.escape(cue)}\b', norm_query):
                    explicit_dosage_form = d_name
                    reasons.append(f"Query explicitly specifies dosage form '{d_name}'.")
                    break
            if explicit_dosage_form:
                break

        if explicit_dosage_form:
            dosage_form = explicit_dosage_form
            explicit_dosage_form_in_query = True
        elif inherited_default_dosage_form:
            dosage_form = inherited_default_dosage_form
            explicit_dosage_form_in_query = False
        else:
            dosage_form = None
            explicit_dosage_form_in_query = False

        # Polysemy check for Amla / Amalaki
        if any("amla" in ing.lower() for ing in detected_ingredients):
            ambiguities.append("Term 'amla' has multiple contextual botanical designations (Phyllanthus emblica / Emblica officinalis).")

        # Polysemy check for Brahmi (Bacopa monnieri vs Centella asiatica)
        if any("brahmi" in ing.lower() for ing in detected_ingredients):
            if "bacopa" in norm_query:
                reasons.append("Query explicitly specifies botanical identity as Bacopa monnieri.")
            elif "centella" in norm_query or "mandukaparni" in norm_query:
                reasons.append("Query explicitly specifies botanical identity as Centella asiatica / Mandukaparni.")
            else:
                ambiguities.append("Term 'brahmi' exhibits regional botanical polysemy (Bacopa monnieri in North India vs. Centella asiatica / Mandukaparni in South India).")

        # Check for multi-ingredient phrase: "contains ashwagandha, turmeric and guduchi"
        m_contains = re.search(r'\b(?:contains|comprising|mixture of|combination of|ingredients?)\s+([a-z\,\s\-and]+)', norm_query)
        if m_contains and not detected_ingredients:
            raw_ing_chunk = m_contains.group(1)
            raw_items = re.split(r'\,|\band\b', raw_ing_chunk)
            for item in raw_items:
                clean_item = item.strip()
                if len(clean_item) > 2 and clean_item not in ["the", "other", "some", "herbs", "plants", "extracts"]:
                    detected_ingredients.append(clean_item.title())

        # General plant / herbal cues
        if any(w in norm_query for w in ["plant", "plants", "herb", "herbal", "polyherbal", "root", "bark", "rhizome", "leaf", "leaves", "seed", "seeds"]):
            plant_signal = True
        if any(w in norm_query for w in ["synthetic", "chemical synthesis", "allopathic", "active pharmaceutical ingredient"]):
            synthetic_signal = True

        # Determine Substance Origin
        origins_found = sum([bool(plant_signal), bool(animal_signal), bool(mineral_signal), bool(synthetic_signal)])
        if synthetic_signal and not (plant_signal or animal_signal or mineral_signal):
            substance_origin = SubstanceOrigin.SYNTHETIC_CHEMICAL
        elif plant_signal and not (animal_signal or mineral_signal or synthetic_signal):
            substance_origin = SubstanceOrigin.BOTANICAL
        elif animal_signal and not (plant_signal or mineral_signal or synthetic_signal):
            substance_origin = SubstanceOrigin.ANIMAL_DERIVED
        elif mineral_signal and not (animal_signal or plant_signal or synthetic_signal):
            substance_origin = SubstanceOrigin.MINERAL_HERBO_MINERAL
        elif origins_found >= 2:
            substance_origin = SubstanceOrigin.MINERAL_HERBO_MINERAL if mineral_signal else SubstanceOrigin.UNKNOWN_MIXED
        else:
            substance_origin = SubstanceOrigin.UNKNOWN_MIXED

        # 5. Process & Novelty Intelligence
        novel_process_signal = (
            any(cue in norm_query for cue in cls.NOVELTY_PROCESS_CUES)
            or (any(n in norm_query for n in ["novel", "new", "improved"]) and any(p in norm_query for p in ["process", "extraction", "synthesis", "method"]))
        )
        has_process_words = any(w in norm_query for w in [
            "extraction", "hydroalcoholic", "decoction", "supercritical", "fermentation",
            "calcination", "purification", "distillation", "fractionation", "nano-formulation",
            "nanoparticle", "synthesis", "method of manufacture", "process for producing",
            "process", "method"
        ])
        if novel_process_signal:
            process_type = ProcessType.POTENTIALLY_NOVEL_PROCESS
            reasons.append("Detected linguistic cues indicating potentially novel or improved process.")
            routing.retrieve_process_standards = True
        elif has_process_words:
            process_type = ProcessType.MODIFIED_PROCESS if any(m in norm_query for m in ["modified", "improved", "altered"]) else ProcessType.KNOWN_TRADITIONAL_PROCESS
            reasons.append(f"Query describes process or extraction method consistent with '{process_type.value}'.")
            routing.retrieve_process_standards = True
        else:
            process_type = ProcessType.UNSPECIFIED_PROCESS

        # 6. Traditional Knowledge Signals
        has_explicit_tk = any(term in norm_query for term in [
            "traditional knowledge", "traditional use", "traditionally known",
            "ancient ayurvedic", "traditional formula", "traditional formulation",
            "ancient ayurvedic text", "described in charaka", "folk knowledge", "indigenous community"
        ])
        has_inferred_tk = is_classical_formulation_name or any(text in norm_query for text in cls.CLASSICAL_TEXTS) or (plant_signal and any(c in norm_query for c in ["classical", "ayurved", "traditional"]))
        if has_explicit_tk:
            tk_signal = TraditionalKnowledgeSignal.EXPLICIT_TRADITIONAL
            reasons.append("Query explicitly references traditional knowledge or community lore.")
            routing.retrieve_traditional_knowledge = True
        elif has_inferred_tk:
            tk_signal = TraditionalKnowledgeSignal.INFERRED_TRADITIONAL
            reasons.append("Query references classical treatise terms or documented traditional substances.")
            routing.retrieve_traditional_knowledge = True
        elif any(term in norm_query for term in ["tradition", "ancient", "ayush"]):
            tk_signal = TraditionalKnowledgeSignal.UNCLEAR_OR_MIXED
        else:
            tk_signal = TraditionalKnowledgeSignal.NO_TK_SIGNAL

        # 7. Intent Classification
        is_patent_inquiry = (
            any(w in norm_query for w in ["patent", "patentable", "patenting", "patentability", "can i patent", "is it patentable", "patent grant", "inventive step", "monopoly", "eligible for patent"])
            or bool(re.search(r'\bsection\s+2\(1\)\(j\)\b', norm_query))
            or bool(re.search(r'\bsection\s+2\b', norm_query) and "invention" in norm_query)
            or ("invention" in norm_query and "section 3" in norm_query)
        )
        if is_patent_inquiry:
            intents.append(UserIntent.PATENTABILITY_INQUIRY)
            routing.retrieve_patentability_criteria = True
            routing.retrieve_patent_exclusions = True
        if any(w in norm_query for w in ["prior art", "traditional knowledge status", "tkdl", "public domain", "anticipated"]):
            intents.append(UserIntent.PRIOR_ART_TK_CONCERN)
            routing.retrieve_traditional_knowledge = True
        if any(w in norm_query for w in ["license", "licensing", "rule 161", "rule 158b", "label", "labelling", "packaging", "sla", "ayush ministry", "gmp", "schedule t"]):
            intents.append(UserIntent.REGULATORY_LICENSING)
            routing.retrieve_regulatory_licensing = True
        if any(w in norm_query for w in ["biodiversity", "nba", "sbb", "biological resource", "benefit sharing", "abs", "prior approval", "section 6", "section 7", "ayush practitioner"]):
            intents.append(UserIntent.BIODIVERSITY_ABS)
            routing.retrieve_biodiversity_abs = True
        if any(w in norm_query for w in ["fee", "fees", "cost", "form 27", "timeline", "procedure"]):
            intents.append(UserIntent.PROCEDURAL_FILING)
        if any(w in norm_query for w in ["compare", "comparison", "comparative", "contrast", "differ", "difference", "versus", " vs ", "under indian law and"]):
            intents.append(UserIntent.COMPARATIVE_CROSS_REGIME)
        if any(w in norm_query for w in ["ayurveda aahar", "food supplement", "dietary supplement", "fssai"]):
            routing.retrieve_food_safety_aahar = True

        if not intents:
            intents.append(UserIntent.GENERAL_INFORMATIONAL)

        # 8. Subject Type Determination
        is_modified = any(m in norm_query for m in ["modified", "modification", "new form", "improved ratio", "altered", "derivative", "isolated fraction"])
        is_proprietary = any(p in norm_query for p in ["patent or proprietary", "patent and proprietary", "p&p", "proprietary formulation", "proprietary ayurvedic"])
        has_classical_text_mention = any(text in norm_query for text in cls.CLASSICAL_TEXTS)

        if is_modified:
            subject_type = SubjectType.MODIFIED_FORMULATION
            reasons.append("Query characterizes the formulation/substance as modified or altered.")
            routing.retrieve_patent_exclusions = True
        elif is_proprietary:
            subject_type = SubjectType.PROPRIETARY_FORMULATION
            reasons.append("Query explicitly refers to patent or proprietary (P&P) Ayurvedic medicine under Rule 158B.")
            routing.retrieve_regulatory_licensing = True
        elif is_classical_formulation_name or has_classical_text_mention:
            subject_type = SubjectType.CLASSICAL_FORMULATION
            reasons.append("Query focuses on a classical Ayurvedic formulation entity or treatise text for regulatory or prior-art verification.")
            routing.retrieve_traditional_knowledge = True
        elif novel_process_signal or (has_process_words and not detected_ingredients and not detected_formulation):
            subject_type = SubjectType.PROCESS_METHOD
            reasons.append("Query primarily concerns a manufacturing, extraction, or synthesis method.")
            routing.retrieve_process_standards = True
        elif detected_ingredients and not detected_formulation and not has_process_words:
            subject_type = SubjectType.SUBSTANCE_INGREDIENT
            reasons.append("Query focuses on an individual botanical or mineral substance rather than a multi-ingredient compound.")
        elif any(r in norm_query for r in ["ayurveda aahar", "food supplement", "cosmetic"]):
            subject_type = SubjectType.REGULATORY_PRODUCT
            reasons.append("Query inquires about designated regulatory product category (Ayurveda Aahar / Cosmetic).")
        elif "formulation" in norm_query or "preparation" in norm_query:
            subject_type = SubjectType.UNSPECIFIED_FORMULATION
            reasons.append("Generic formulation indicated without specified ingredients or classical reference.")
            missing_info.append("Specific ingredient list or classical reference not provided.")
        elif any(g in norm_query for g in ["section 3", "patents act", "drugs and cosmetics", "wipo", "nagoya", "trips"]):
            subject_type = SubjectType.GENERAL_INQUIRY
            reasons.append("Statutory or regime-level inquiry without a concrete formulation subject.")
        else:
            subject_type = SubjectType.UNKNOWN
            reasons.append("Query does not contain clear formulation or substance identifiers.")

        # Missing Information Tracking
        if subject_type in [SubjectType.CLASSICAL_FORMULATION, SubjectType.PROPRIETARY_FORMULATION, SubjectType.MODIFIED_FORMULATION, SubjectType.UNSPECIFIED_FORMULATION]:
            if not detected_ingredients:
                missing_info.append("Exact constituent ingredient ratios or full ingredient list absent from query.")
            if not dosage_form:
                missing_info.append("Finished dosage form (e.g. churna, vati, avaleha) not specified.")

        # 9. Confidence Assessment
        # Distinguish explicit query cues from dictionary-inferred defaults:
        # An inherited default dosage form is NOT counted as an independent explicit signal.
        signal_count = sum([
            bool(detected_formulation),
            bool(detected_ingredients),
            explicit_dosage_form_in_query,
            is_classical_formulation_name,
            novel_process_signal,
            has_explicit_tk,
            len(intents) > 0 and intents[0] != UserIntent.GENERAL_INFORMATIONAL
        ])

        if subject_type == SubjectType.UNKNOWN:
            overall_confidence = ConfidenceTier.UNKNOWN
            conf_score = 0.0
        elif signal_count >= 3 and not ambiguities:
            overall_confidence = ConfidenceTier.HIGH
            conf_score = 0.90
        elif signal_count >= 1:
            overall_confidence = ConfidenceTier.MEDIUM
            conf_score = 0.70
        else:
            overall_confidence = ConfidenceTier.LOW
            conf_score = 0.40

        # Suggested Jurisdictions
        suggested_jurs: List[str] = []
        if any(w in norm_query for w in ["wipo", "gratk", "nagoya", "trips", "international", "global", "cbd"]):
            suggested_jurs.append("international")
        if any(w in norm_query for w in ["india", "indian", "patents act", "section 3", "rule 161", "rule 158b", "nba", "sbb", "sla", "ayush", "fssai"]):
            suggested_jurs.append("national")
        if not suggested_jurs:
            suggested_jurs = [jurisdiction] if jurisdiction in ["national", "international", "comparative"] else ["national"]
        routing.jurisdictions_suggested = suggested_jurs

        return FormulationIntelligence(
            query_text=raw_query,
            normalized_text=norm_query,
            subject_type=subject_type,
            subject_confidence=overall_confidence,
            formulation_name=detected_formulation,
            alternative_names=alt_names,
            ingredients=detected_ingredients,
            ingredient_count=len(detected_ingredients),
            substance_origin=substance_origin,
            dosage_form=dosage_form,
            preparation_method=None,
            process_type=process_type,
            novel_process_signal=novel_process_signal,
            traditional_knowledge_signal=tk_signal,
            user_intents=intents,
            plant_origin_signal=plant_signal,
            animal_origin_signal=animal_signal,
            mineral_origin_signal=mineral_signal,
            microbial_origin_signal=False,
            synthetic_chemical_signal=synthetic_signal,
            food_or_ayush_product_signal=any(f in norm_query for f in ["food", "ayurveda aahar", "supplement", "cosmetic"]),
            patent_inquiry_signal=UserIntent.PATENTABILITY_INQUIRY in intents,
            regulatory_inquiry_signal=UserIntent.REGULATORY_LICENSING in intents,
            biodiversity_signal=UserIntent.BIODIVERSITY_ABS in intents or plant_signal,
            abs_signal=UserIntent.BIODIVERSITY_ABS in intents,
            jurisdictions_relevant=suggested_jurs,
            classification_reasons=reasons,
            ambiguities=ambiguities,
            missing_information=missing_info,
            routing_hints=routing,
            overall_confidence=overall_confidence,
            confidence_score=conf_score,
            classifier_version="v1.0-deterministic"
        )

    # Legacy method retained solely for backward API signature compatibility
    @classmethod
    def classify(cls, inputs: FormulationInput) -> ClassificationResult:
        """
        DEPRECATED: Legacy stub retained only for backward API signature compatibility.
        Does not perform active legal adjudication. Use classify_query() for production pipeline.
        """
        category_id = "classical" if inputs.is_first_schedule else ("proprietary" if inputs.has_proprietary_formulation else "general_ayush")
        cat_name = "Classical / Generic Ayurvedic Medicine" if inputs.is_first_schedule else (
            "Patent or Proprietary (P&P) Ayurvedic Medicine" if inputs.has_proprietary_formulation else "General AYUSH Formulation"
        )
        return ClassificationResult(
            category_id=category_id,
            category_name=cat_name,
            statutory_definition="Ayurvedic medicine category per Drugs and Cosmetics Act First Schedule or Rule 158B.",
            governing_regime="Drugs and Cosmetics Act, 1940",
            licensing_authority="State Licensing Authority (SLA)",
            ip_posture={"patent_eligibility": "Requires independent statutory examination under Patents Act Section 3 and Section 2(1)(j)."},
            abs_posture={"status": "Subject to Biological Diversity Act requirements and exemptions."},
            statutory_citations=["Drugs and Cosmetics Act, 1940"]
        )

# Module-level singleton
classifier_engine = FormulationClassifierEngine()
