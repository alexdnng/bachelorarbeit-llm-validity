# =====================================================
# MODEL CONFIGURATION
# =====================================================


OPENAI_MODELS = [
    #"gpt-4.1",
    "gpt-4.1-mini"
]

OLLAMA_MODELS = [
    "llama3.1:8b"
]

JUDGE_MODEL="gpt-4.1-mini"

# Select active model group
#MODELS = OLLAMA_MODELS
MODELS = OPENAI_MODELS


# =====================================================
# EXPERIMENT SETTINGS
# =====================================================

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]

REASONING_MODES = [
    "direct",
    "think",
    "cot"
]

TOP_P_VALUES = [1.0]

samplesize = 20

MAX_TOKENS = 100


N_RUNS_PER_SETTING = 3

# =====================================================
# DIGITAL TWIN FEATURE CONFIGURATION
# =====================================================

TWIN_FEATURE_COLUMNS = [
    "score_needforcognition",
    "score_agency",
    "score_minimalism",
    "score_BES",
    "score_GREEN",
    "score_socialdesirability",
    "score_anxiety",
    "score_HI",
    "score_VI",
    "score_HC",
    "score_VC",
    "score_RFS",
    "score_ST-TW",
    "score_depression",
    "score_CNFU-S",
    "score_selfmonitor",
    "score_SCC",
    "score_needforclosure",
    "score_maximization"
]

FEATURE_LABELS = {
    "score_needforcognition": "Need for Cognition scale (enjoys thinking deeply, analyzing information, and engaging with complex ideas)",
    "score_agency": "Agentic vs. Communal Values scale (prioritizes personal achievement, independence, and success versus cooperation, care, and social relationships)",
    "score_minimalism": "Consumer Minimalism scale (prefers simplicity, reduced material consumption, and owning only what is necessary)",
    "score_BES": "Basic Empathy scale (understands and shares the feelings and emotional experiences of others)",
    "score_GREEN": "Green Values scale (cares about environmental protection, sustainability, and ecological responsibility)",
    "score_socialdesirability": "Social Desirability scale (tends to present oneself in a socially acceptable and favorable way)",
    "score_anxiety": "Anxiety scale (experiences worry, nervousness, uncertainty, and emotional tension more frequently)",
    "score_HI": "Horizontal Individualism (values independence and self-reliance while viewing others as equals)",
    "score_VI": "Vertical Individualism (values independence, competition, status, and standing out from others)",
    "score_HC": "Horizontal Collectivism (values cooperation, group harmony, and equality within social groups)",
    "score_VC": "Vertical Collectivism (values loyalty to groups, fulfilling duties, and respecting hierarchy and authority)",
    "score_RFS": "Regulatory Focus scale (approaches goals by seeking growth and gains or by avoiding risks and losses)",
    "score_ST-TW": "Tightwads vs. Spendthrift scale (tendency to be cautious and reluctant versus willing and spontaneous when spending money)",
    "score_depression": "Depression scale (experiences sadness, low motivation, hopelessness, and reduced enjoyment more frequently)",
    "score_CNFU-S": "Need for Uniqueness scale (desires to be different, distinctive, and stand out from others)",
    "score_selfmonitor": "Self-Monitoring scale (adjusts behavior and self-presentation to fit different social situations)",
    "score_SCC": "Self-Concept Clarity scale (has a clear, stable, and well-defined understanding of personal identity)",
    "score_needforclosure": "Need for Closure scale (prefers certainty, structure, predictability, and clear answers over ambiguity)",
    "score_maximization": "Maximization scale (seeks the best possible option and carefully compares alternatives before deciding)"
}

GROUND_TRUTH_COLUMNS = [
    "score_extraversion",
    "score_agreeableness",
    "score_conscientiousness",
    "score_openness",
    "score_neuroticism"
]

QUESTIONS = [
        "Do you see yourself as someone who is reserved?",
        "Do you see yourself as someone who is generally trusting?",
        "Do you see yourself as someone who tends to be lazy?",
        "Do you see yourself as someone who is relaxed and handles stress well?",
        "Do you see yourself as someone who has few artistic interests?",
        "Do you see yourself as someone who is outgoing and sociable?",
        "Do you see yourself as someone who tends to find fault with others?",
        "Do you see yourself as someone who does a thorough job?",
        "Do you see yourself as someone who gets nervous easily?",
        "Do you see yourself as someone who has an active imagination?"
    ]