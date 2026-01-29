# fusion_engine.py
from typing import Dict, List, Literal, TypedDict
import pickle
import os
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
from logger_config import get_logger

logger = get_logger("fusion_engine")


# ---- Type Definitions ----

MotionLevel = Literal["NONE", "LOW", "HIGH"]
HeatPresence = Literal["NORMAL", "LOW"]
BreathingStatus = Literal["YES", "NO"]
ConfidenceLevel = Literal["LOW", "MEDIUM", "HIGH"]
Responsiveness = Literal["RESPONSIVE", "WEAK_RESPONSE", "UNRESPONSIVE", "UNKNOWN"]
VitalSigns = Literal["STABLE_SIGNS", "UNCERTAIN_SIGNS", "WEAK_SIGNS", "UNKNOWN"]


class SensorData(TypedDict):
    human_detected: bool
    motion_level: MotionLevel
    heat_presence: HeatPresence
    breathing_detected: BreathingStatus


class VictimState(TypedDict):
    presence_confirmed: bool
    responsiveness: Responsiveness
    vital_signs: VitalSigns
    confidence_level: ConfidenceLevel
    fusion_explanation: List[str]


# ---- Fusion Logic ----

# Initialize model globally
_model = None
_encoder_map = {
    "responsiveness": {"RESPONSIVE": 2, "WEAK_RESPONSE": 1, "UNRESPONSIVE": 0, "UNKNOWN": -1},
    "vital_signs": {"STABLE_SIGNS": 2, "UNCERTAIN_SIGNS": 1, "WEAK_SIGNS": 0, "UNKNOWN": -1},
    "confidence": {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
}
MODEL_PATH = os.path.join(os.path.dirname(__file__), "fusion_model.pkl")


def _create_training_data():
    """Load training data from CSV dataset"""
    dataset_path = os.path.join(os.path.dirname(__file__), "sensor_training_data.csv")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    
    X = []
    y = []
    
    # Load CSV dataset
    with open(dataset_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            features = [
                int(row['human_detected']),
                int(row['motion_level']),
                int(row['heat_presence']),
                int(row['breathing_detected'])
            ]
            label = int(row['confidence_level'])
            X.append(features)
            y.append(label)
    
    logger.info(f"Loaded {len(X)} training samples from sensor_training_data.csv")
    
    return np.array(X), np.array(y)


def save_model():
    """Save trained model to disk"""
    global _model
    if _model is None:
        logger.error("Attempted to save model but no model is trained")
        raise ValueError("No trained model to save")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(_model, f)
    logger.info(f"Model saved to {MODEL_PATH}")


def load_model():
    """Load model from disk if it exists"""
    global _model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
        logger.info(f"Model loaded from {MODEL_PATH}")
        return _model
    logger.debug("No saved model found, will train new model")
    return None


def train_model(save=True):
    """Train the fusion model and optionally save it"""
    global _model
    logger.info("Starting model training...")
    X, y = _create_training_data()
    _model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    _model.fit(X, y)
    logger.info(f"Model trained with {len(X)} samples")
    
    if save:
        save_model()
    
    return _model


def _encode_sensor_data(sensor_data: SensorData) -> List[int]:
    """Convert sensor data to numeric features for model"""
    motion_map = {"NONE": 0, "LOW": 1, "HIGH": 2}
    heat_map = {"LOW": 0, "NORMAL": 1}
    
    return [
        1 if sensor_data["human_detected"] else 0,
        motion_map[sensor_data["motion_level"]],
        heat_map[sensor_data["heat_presence"]],
        1 if sensor_data["breathing_detected"] == "YES" else 0
    ]


def validate_sensor_data(sensor_data: SensorData) -> tuple[bool, str]:
    """
    Validate sensor data for correctness.
    
    Returns:
        tuple of (is_valid, error_message)
    """
    # Check if required keys exist
    required_keys = {"human_detected", "motion_level", "heat_presence", "breathing_detected"}
    if not all(key in sensor_data for key in required_keys):
        missing = required_keys - set(sensor_data.keys())
        error_msg = f"Missing required fields: {missing}"
        logger.warning(f"Validation failed: {error_msg}")
        return False, error_msg
    
    # Validate data types
    if not isinstance(sensor_data["human_detected"], bool):
        error_msg = "human_detected must be boolean"
        logger.warning(f"Validation failed: {error_msg}")
        return False, error_msg
    
    if sensor_data["motion_level"] not in ["NONE", "LOW", "HIGH"]:
        error_msg = f"motion_level must be NONE/LOW/HIGH, got {sensor_data['motion_level']}"
        logger.warning(f"Validation failed: {error_msg}")
        return False, error_msg
    
    if sensor_data["heat_presence"] not in ["LOW", "NORMAL"]:
        error_msg = f"heat_presence must be LOW/NORMAL, got {sensor_data['heat_presence']}"
        logger.warning(f"Validation failed: {error_msg}")
        return False, error_msg
    
    if sensor_data["breathing_detected"] not in ["YES", "NO"]:
        error_msg = f"breathing_detected must be YES/NO, got {sensor_data['breathing_detected']}"
        logger.warning(f"Validation failed: {error_msg}")
        return False, error_msg
    
    logger.debug("Sensor data validation passed")
    return True, ""


def evaluate_model_performance(X, y):
    """
    Evaluate model performance using cross-validation and metrics.
    
    Returns:
        Dictionary with evaluation metrics
    """
    global _model
    
    if _model is None:
        return {"error": "Model not trained"}
    
    # Cross-validation scores
    cv_scores = cross_val_score(_model, X, y, cv=5, scoring='accuracy')
    
    # Predictions for detailed metrics
    y_pred = _model.predict(X)
    
    # Calculate metrics
    precision = precision_score(y, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    
    return {
        "cross_validation_scores": cv_scores.tolist(),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": conf_matrix.tolist(),
        "accuracy": float(cv_scores.mean())
    }


def fuse_signals(sensor_data: SensorData) -> VictimState:
    global _model
    
    # Validate input data
    is_valid, error_msg = validate_sensor_data(sensor_data)
    if not is_valid:
        logger.error(f"Invalid sensor data: {error_msg}")
        raise ValueError(f"Invalid sensor data: {error_msg}")
    
    # Try to load saved model first
    if _model is None:
        load_model()
    
    # If still no model, train it
    if _model is None:
        train_model(save=True)
    
    logger.debug(f"Processing sensor data: {sensor_data}")
    
    victim_state: VictimState = {
        "presence_confirmed": False,
        "responsiveness": "UNKNOWN",
        "vital_signs": "UNKNOWN",
        "confidence_level": "LOW",
        "fusion_explanation": []
    }

    # Human presence
    if not sensor_data["human_detected"]:
        victim_state["fusion_explanation"].append(
            "No human presence detected by sensors"
        )
        logger.info("No human presence detected")
        return victim_state

    victim_state["presence_confirmed"] = True
    victim_state["fusion_explanation"].append(
        "Human presence confirmed"
    )
    logger.info("Human presence confirmed - analyzing vitals")

    # Use trained model to predict confidence level
    features = _encode_sensor_data(sensor_data)
    confidence_pred = _model.predict([features])[0]
    
    confidence_map = {0: "LOW", 1: "MEDIUM", 2: "HIGH"}
    victim_state["confidence_level"] = confidence_map[confidence_pred]
    
    # Determine responsiveness and vital signs for explanation
    motion: MotionLevel = sensor_data["motion_level"]

    if motion == "HIGH":
        victim_state["responsiveness"] = "RESPONSIVE"
        victim_state["fusion_explanation"].append(
            "High motion responsiveness detected"
        )
    elif motion == "LOW":
        victim_state["responsiveness"] = "WEAK_RESPONSE"
        victim_state["fusion_explanation"].append(
            "Low motion responsiveness detected"
        )
    else:
        victim_state["responsiveness"] = "UNRESPONSIVE"
        victim_state["fusion_explanation"].append(
            "No motion responsiveness detected"
        )

    # Vital signs (non-medical)
    heat: HeatPresence = sensor_data["heat_presence"]
    breathing: BreathingStatus = sensor_data["breathing_detected"]

    if heat == "NORMAL" and breathing == "YES":
        victim_state["vital_signs"] = "STABLE_SIGNS"
        victim_state["fusion_explanation"].append(
            "Normal heat and breathing detected"
        )
    elif heat == "NORMAL" and breathing == "NO":
        victim_state["vital_signs"] = "UNCERTAIN_SIGNS"
        victim_state["fusion_explanation"].append(
            "Heat detected but breathing not confirmed"
        )
    else:
        victim_state["vital_signs"] = "WEAK_SIGNS"
        victim_state["fusion_explanation"].append(
            "Low heat or no breathing detected"
        )

    victim_state["fusion_explanation"].append(
        f"Model predicted confidence: {victim_state['confidence_level']}"
    )

    return victim_state


# ---- Test Cases ----

def run_tests():
    """Run test cases for the fusion engine"""
    
    test_cases = [
        {
            "name": "Responsive victim with stable vital signs",
            "input": {
                "human_detected": True,
                "motion_level": "HIGH",
                "heat_presence": "NORMAL",
                "breathing_detected": "YES"
            },
            "expected_confidence": "HIGH"
        },
        {
            "name": "Weak response, uncertain vital signs",
            "input": {
                "human_detected": True,
                "motion_level": "LOW",
                "heat_presence": "NORMAL",
                "breathing_detected": "NO"
            },
            "expected_confidence": "MEDIUM"
        },
        {
            "name": "No response, weak vital signs",
            "input": {
                "human_detected": True,
                "motion_level": "NONE",
                "heat_presence": "LOW",
                "breathing_detected": "NO"
            },
            "expected_confidence": "MEDIUM"
        },
        {
            "name": "No human detected",
            "input": {
                "human_detected": False,
                "motion_level": "NONE",
                "heat_presence": "LOW",
                "breathing_detected": "NO"
            },
            "expected_confidence": "LOW"
        },
        {
            "name": "Responsive with weak vital signs",
            "input": {
                "human_detected": True,
                "motion_level": "HIGH",
                "heat_presence": "LOW",
                "breathing_detected": "NO"
            },
            "expected_confidence": "HIGH"
        }
    ]
    
    print("=" * 70)
    print("FUSION ENGINE TEST CASES")
    print("=" * 70)
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test['name']}")
        print("-" * 70)
        
        result = fuse_signals(test['input'])
        
        print(f"  Input: {test['input']}")
        print(f"\n  Results:")
        print(f"    Presence Confirmed: {result['presence_confirmed']}")
        print(f"    Responsiveness: {result['responsiveness']}")
        print(f"    Vital Signs: {result['vital_signs']}")
        print(f"    Confidence Level: {result['confidence_level']}")
        print(f"\n  Fusion Explanation:")
        for explanation in result['fusion_explanation']:
            print(f"    - {explanation}")
        
        status = "PASS" if result['confidence_level'] == test['expected_confidence'] else "FAIL"
        print(f"\n  Expected Confidence: {test['expected_confidence']} [{status}]")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # First run: train and save model
    print("Training model...")
    X, y = _create_training_data()
    train_model(save=True)
    print("Model trained and saved successfully!\n")
    
    # Evaluate model performance
    print("="*70)
    print("MODEL EVALUATION METRICS")
    print("="*70)
    metrics = evaluate_model_performance(X, y)
    
    print(f"\nCross-Validation Results:")
    print(f"  Mean Accuracy: {metrics['cv_mean']:.4f}")
    print(f"  Std Dev: {metrics['cv_std']:.4f}")
    print(f"  Fold Scores: {[f'{s:.4f}' for s in metrics['cross_validation_scores']]}")
    
    print(f"\nDetailed Metrics:")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1-Score: {metrics['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    conf = metrics['confusion_matrix']
    print(f"  {conf}")
    
    print("\n" + "="*70 + "\n")
    
    run_tests()
