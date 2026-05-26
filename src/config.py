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

# Select active model group
#MODELS = OLLAMA_MODELS
MODELS = OPENAI_MODELS


# =====================================================
# EXPERIMENT SETTINGS
# =====================================================

TEMPERATURES = [0.1, 0.3, 0.5, 0.7, 0.9]

REASONING_MODES = [
    "direct",
    "cot",
    "uncertain"
]

TOP_P_VALUES = [1.0]

samplesize = 25

MAX_TOKENS = 100


N_RUNS_PER_SETTING = 1

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
    "score_needforcognition": "Need for cognition scale",
    "score_agency": "Agentic vs. Communal Values scale",
    "score_minimalism": "Consumer Minimalism scale",
    "score_BES": "Basic Empathy scale",
    "score_GREEN": "Green values scale",
    "score_socialdesirability": "Social Desirability scale",
    "score_anxiety": "Anxiety scale",
    "score_HI": "Horizontal Individualism",
    "score_VI": "Vertical Individualism",
    "score_HC": "Horizontal Collectivism",
    "score_VC": "Vertical Collectivism",
    "score_RFS": "Regulatory Focus scale",
    "score_ST-TW": "Tightwads vs. Spendthrift scale",
    "score_depression": "Depression scale",
    "score_CNFU-S": "Need for uniqueness scale",
    "score_selfmonitor": "Self-monitoring scale",
    "score_SCC": "Self-concept clarity scale",
    "score_needforclosure": "Need for closure scale",
    "score_maximization": "Maximization scale"
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