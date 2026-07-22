# backend/services/pdi_risk_model_service.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sqlalchemy.orm import Session
from typing import Dict
import logging

from backend.models.feedback_model import AggregatedPerformance
from backend.models.employee_model import EmployeePerformanceView

logger = logging.getLogger(__name__)


class PDIRiskModelService:
    """
    Supervised ML component for Step 1 (Opinion Dynamics / Power Distance):
    predicts an employee-quarter's power-distance-influence risk category
    from demographic + role features, using the existing divergence_score
    thresholds as ground-truth labels.

    This complements (does not replace) the formula-based divergence_score —
    the formula generates the label, the model demonstrates that risk is
    predictable from observable features, which is itself a research finding.
    """

    MODEL_PATH = Path("models/pdi_risk_model.pkl")
    ENCODER_PATH = Path("models/pdi_risk_encoders.pkl")

    FEATURE_COLS_NUMERIC = ['years_of_experience']
    FEATURE_COLS_CATEGORICAL = ['gender', 'age_group', 'department', 'job_role', 'ethnicity']

    def __init__(self, db: Session):
        self.db = db
        self.model = None
        self.encoders = {}
        self.MODEL_PATH.parent.mkdir(exist_ok=True)

    def _build_training_frame(self) -> pd.DataFrame:
        agg_rows = self.db.query(AggregatedPerformance).all()
        agg_df = pd.DataFrame([{
            'employee_id': r.employee_id,
            'divergence_score': r.divergence_score
        } for r in agg_rows]).groupby('employee_id', as_index=False).mean(numeric_only=True)

        emp_rows = self.db.query(EmployeePerformanceView).all()
        emp_df = pd.DataFrame([{
            'employee_id': e.employee_id,
            **{c: getattr(e, c, None) for c in self.FEATURE_COLS_NUMERIC + self.FEATURE_COLS_CATEGORICAL}
        } for e in emp_rows]).drop_duplicates(subset='employee_id')

        df = emp_df.merge(agg_df, on='employee_id', how='inner').dropna()

        # Label: same thresholds you already use in FeedbackService._get_divergence_level
        def label(score):
            if score < 0.10:
                return 'Low'
            elif score < 0.20:
                return 'Moderate'
            else:
                return 'High'
        df['pdi_risk_label'] = df['divergence_score'].apply(label)
        return df

    def train(self) -> Dict:
        df = self._build_training_frame()
        if df.empty or df['pdi_risk_label'].nunique() < 2:
            return {'status': 'error', 'message': 'Insufficient data or label variance to train'}

        X = df[self.FEATURE_COLS_NUMERIC + self.FEATURE_COLS_CATEGORICAL].copy()
        y = df['pdi_risk_label']

        self.encoders = {}
        for col in self.FEATURE_COLS_CATEGORICAL:
            enc = LabelEncoder()
            X[col] = enc.fit_transform(X[col].astype(str))
            self.encoders[col] = enc

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)

        joblib.dump(self.model, self.MODEL_PATH)
        joblib.dump(self.encoders, self.ENCODER_PATH)

        feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        return {
            'status': 'success',
            'n_samples': len(df),
            'label_distribution': df['pdi_risk_label'].value_counts().to_dict(),
            'accuracy': round(float(accuracy), 4),
            'macro_f1': round(float(f1), 4),
            'classification_report': report,
            'feature_importance': [{'feature': f, 'importance': round(float(i), 4)} for f, i in top_features]
        }

    def predict(self, employee_features: Dict) -> Dict:
        if self.model is None:
            if self.MODEL_PATH.exists():
                self.model = joblib.load(self.MODEL_PATH)
                self.encoders = joblib.load(self.ENCODER_PATH)
            else:
                raise ValueError("Model not trained yet")

        X = pd.DataFrame([employee_features])
        for col in self.FEATURE_COLS_CATEGORICAL:
            if col in self.encoders:
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except ValueError:
                    X[col] = 0  # unseen category fallback

        X = X[self.FEATURE_COLS_NUMERIC + self.FEATURE_COLS_CATEGORICAL]
        pred = self.model.predict(X)[0]
        proba = dict(zip(self.model.classes_, self.model.predict_proba(X)[0]))
        return {'predicted_risk': pred, 'probabilities': {k: round(float(v), 3) for k, v in proba.items()}}