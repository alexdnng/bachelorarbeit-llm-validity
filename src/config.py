# =====================================================
# MODEL CONFIGURATION
# =====================================================

OPENAI_MODELS = [
    "gpt-4.1",
    "gpt-4.1-mini"
]

OLLAMA_MODELS = [
    "llama3.1:8b"
]

# Select active model group
MODELS = OLLAMA_MODELS
#MODELS = OPENAI_MODELS


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

samplesize = 5

MAX_TOKENS = 300

N_RUNS_PER_SETTING = 5

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