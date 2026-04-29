import re

TRAITS = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness"
]

def parse_traits(text):
    result = {}
    
    for trait in TRAITS:
        match = re.search(rf"{trait}:\s*([0-9\.]+)", text)
        if match:
            result[trait] = float(match.group(1))
        else:
            result[trait] = None
    
    return result


def compute_mae(pred, true):
    errors = []
    
    for trait in TRAITS:
        if pred[trait] is not None and trait in true:
            errors.append(abs(pred[trait] - true[trait]))
    
    if len(errors) == 0:
        return None
    
    return sum(errors) / len(errors)