import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from backend.models.history_model import PerformanceHistory
from backend.models.prediction_model import PerformancePrediction
from backend.models.metadata_model import ModelMetadata
from backend.models.result_model import PerformanceResult

class PredictionService:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.rf_model = None
        self.gb_model = None
        self.scaler = None
        self.encoders = {}
        self.feature_names = []
        self.rf_weight = 0.5
        self.gb_weight = 0.5

        # Model paths
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)

        self.rf_model_path = self.model_dir / "rf_model.pkl"
        self.gb_model_path = self.model_dir / "gb_model.pkl"
        self.scaler_path = self.model_dir / "scaler.pkl"
        self.encoder_path = self.model_dir / "encoders.pkl"
        self.feature_path = self.model_dir / "feature_names.pkl"

        self._load_models()

    def _load_models(self):
        """Load trained models"""
        try:
            if self.rf_model_path.exists():
                self.rf_model = joblib.load(self.rf_model_path)
                print("✅ RF model loaded")
            if self.gb_model_path.exists():
                self.gb_model = joblib.load(self.gb_model_path)
                print("✅ GB model loaded")
            if self.scaler_path.exists():
                self.scaler = joblib.load(self.scaler_path)
                print("✅ Scaler loaded")
            if self.encoder_path.exists():
                self.encoders = joblib.load(self.encoder_path)
                print("✅ Encoders loaded")
            if self.feature_path.exists():
                self.feature_names = joblib.load(self.feature_path)
                print("✅ Feature names loaded")
        except Exception as e:
            print(f"⚠️ Could not load models: {e}")

    def train_models(self) -> Dict:
        """Train both Random Forest and Gradient Boosting models"""
        print("\n🔧 Training Hybrid Model...")

        # Get historical data
        records = self.db.query(PerformanceHistory).order_by(
            PerformanceHistory.employee_id,
            PerformanceHistory.period_year,
            PerformanceHistory.period_quarter
        ).all()

        if not records:
            raise Exception("No historical data available for training")

        print(f"📊 Found {len(records)} historical records")

        # Prepare data
        data = []
        for r in records:
            data.append({
                'employee_id': r.employee_id,
                'period_year': r.period_year,
                'period_quarter': r.period_quarter,
                'job_role': r.job_role or 'Unknown',
                'years_of_experience': r.years_of_experience or 0,
                'department': r.department or 'Unknown',
                'duration_weeks': r.duration_weeks or 0,
                'relative_effort': r.relative_effort or 0,
                'team_size': r.team_size or 1,
                'project_complexity': r.project_complexity or 'Medium',
                'rework_count': r.rework_count or 0,
                'no_pay_leave': r.no_pay_leave or 0,
                'blockers': r.blockers or 0,
                'punctuality': r.punctuality or 3,
                'problem_solving': r.problem_solving or 3,
                'leadership': r.leadership or 3,
                'collaboration': r.collaboration or 3,
                'communication': r.communication or 3,
                'deadline_adherence_rate': r.deadline_adherence_rate or 0,
                'avg_response_time': r.avg_response_time or 0,
                'completed_storypoint_ratio': r.completed_storypoint_ratio or 0,
                'completed_story_points': r.completed_story_points or 0,
                'assigned_story_points': r.assigned_story_points or 0,
                'metric_1_value': r.metric_1_value or 0,
                'metric_2_value': r.metric_2_value or 0,
                'metric_3_value': r.metric_3_value or 0,
                'metric_4_value': r.metric_4_value or 0,
                'metric_5_value': r.metric_5_value or 0,
                'metric_6_value': r.metric_6_value or 0,
                'metric_7_value': r.metric_7_value or 0
            })

        df = pd.DataFrame(data)

        # Calculate performance score
        df['performance_score'] = (
                df['deadline_adherence_rate'] * 0.25 +
                df['completed_storypoint_ratio'] * 100 * 0.25 +
                df['punctuality'] * 5 * 0.10 +
                df['problem_solving'] * 5 * 0.10 +
                df['leadership'] * 5 * 0.10 +
                df['collaboration'] * 5 * 0.10 +
                df['communication'] * 5 * 0.10
        )

        # Create target bands
        df['performance_band'] = pd.cut(
            df['performance_score'],
            bins=[-1, 40, 70, 100],
            labels=['Low', 'Medium', 'High']
        )

        # Prepare features
        feature_cols = [
            'job_role', 'years_of_experience', 'department',
            'duration_weeks', 'relative_effort', 'team_size',
            'project_complexity', 'rework_count', 'no_pay_leave',
            'blockers', 'punctuality', 'problem_solving',
            'leadership', 'collaboration', 'communication',
            'deadline_adherence_rate', 'avg_response_time',
            'completed_storypoint_ratio', 'completed_story_points',
            'assigned_story_points'
        ]

        X = df[feature_cols].copy()
        y = df['performance_band'].copy()

        # Encode categorical variables
        categorical_cols = ['job_role', 'department', 'project_complexity']
        self.encoders = {}

        for col in categorical_cols:
            if col in X.columns:
                encoder = LabelEncoder()
                X[col] = encoder.fit_transform(X[col].astype(str))
                self.encoders[col] = encoder

        # Handle missing values
        X = X.fillna(X.median())

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Save feature names
        self.feature_names = X.columns.tolist()
        joblib.dump(self.feature_names, self.feature_path)
        joblib.dump(self.encoders, self.encoder_path)
        joblib.dump(self.scaler, self.scaler_path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"📊 Training data: {len(X_train)} samples, Test data: {len(X_test)} samples")

        # Train Random Forest
        print("🌳 Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        rf_pred = self.rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        rf_f1 = f1_score(y_test, rf_pred, average='macro')
        print(f"  RF Accuracy: {rf_accuracy:.4f}, F1: {rf_f1:.4f}")

        # Train Gradient Boosting
        print("📈 Training Gradient Boosting...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            min_samples_leaf=5,
            learning_rate=0.1,
            random_state=42
        )
        self.gb_model.fit(X_train, y_train)
        gb_pred = self.gb_model.predict(X_test)
        gb_accuracy = accuracy_score(y_test, gb_pred)
        gb_f1 = f1_score(y_test, gb_pred, average='macro')
        print(f"  GB Accuracy: {gb_accuracy:.4f}, F1: {gb_f1:.4f}")

        # Calculate ensemble weights
        total_f1 = rf_f1 + gb_f1
        self.rf_weight = rf_f1 / total_f1 if total_f1 > 0 else 0.5
        self.gb_weight = gb_f1 / total_f1 if total_f1 > 0 else 0.5

        print(f"⚖️ Ensemble Weights: RF={self.rf_weight:.3f}, GB={self.gb_weight:.3f}")

        # Save models
        joblib.dump(self.rf_model, self.rf_model_path)
        joblib.dump(self.gb_model, self.gb_model_path)

        # Hybrid prediction
        rf_proba = self.rf_model.predict_proba(X_test)
        gb_proba = self.gb_model.predict_proba(X_test)
        hybrid_proba = (self.rf_weight * rf_proba + self.gb_weight * gb_proba)
        hybrid_pred_idx = np.argmax(hybrid_proba, axis=1)
        hybrid_pred = self.rf_model.classes_[hybrid_pred_idx]
        hybrid_accuracy = accuracy_score(y_test, hybrid_pred)
        hybrid_f1 = f1_score(y_test, hybrid_pred, average='macro')

        print(f"🤝 Hybrid Accuracy: {hybrid_accuracy:.4f}, F1: {hybrid_f1:.4f}")

        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.rf_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]

        print(f"\n🏆 Top 10 Important Features:")
        for feat, imp in top_features:
            print(f"  - {feat}: {imp:.4f}")

        # Save model metadata
        metadata = ModelMetadata(
            model_name=f"hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            algorithm_type="Hybrid",
            accuracy=hybrid_accuracy,
            f1_score=hybrid_f1,
            feature_importance=json.dumps(top_features),
            feature_names=json.dumps(self.feature_names),
            is_active=1
        )
        self.db.add(metadata)
        self.db.commit()

        print("✅ Model training complete!")

        return {
            'accuracy': hybrid_accuracy,
            'f1_score': hybrid_f1,
            'rf_accuracy': rf_accuracy,
            'gb_accuracy': gb_accuracy,
            'top_features': [{'feature': f, 'importance': i} for f, i in top_features[:5]]
        }

    def predict_employee(self, employee_id: str) -> Optional[Dict]:
        """Predict next quarter performance for an employee"""
        if self.rf_model is None or self.gb_model is None:
            self._load_models()

        if self.rf_model is None or self.gb_model is None:
            raise Exception("Models not trained")

        # Get latest data
        latest = self.db.query(PerformanceHistory).filter_by(
            employee_id=employee_id
        ).order_by(
            PerformanceHistory.period_year.desc(),
            PerformanceHistory.period_quarter.desc()
        ).first()

        if not latest:
            return None

        # Prepare features
        feature_dict = {
            'job_role': latest.job_role or 'Unknown',
            'years_of_experience': latest.years_of_experience or 0,
            'department': latest.department or 'Unknown',
            'duration_weeks': latest.duration_weeks or 0,
            'relative_effort': latest.relative_effort or 0,
            'team_size': latest.team_size or 1,
            'project_complexity': latest.project_complexity or 'Medium',
            'rework_count': latest.rework_count or 0,
            'no_pay_leave': latest.no_pay_leave or 0,
            'blockers': latest.blockers or 0,
            'punctuality': latest.punctuality or 3,
            'problem_solving': latest.problem_solving or 3,
            'leadership': latest.leadership or 3,
            'collaboration': latest.collaboration or 3,
            'communication': latest.communication or 3,
            'deadline_adherence_rate': latest.deadline_adherence_rate or 0,
            'avg_response_time': latest.avg_response_time or 0,
            'completed_storypoint_ratio': latest.completed_storypoint_ratio or 0,
            'completed_story_points': latest.completed_story_points or 0,
            'assigned_story_points': latest.assigned_story_points or 0
        }

        # Convert to DataFrame
        X = pd.DataFrame([feature_dict])

        # Encode categorical columns
        for col in ['job_role', 'department', 'project_complexity']:
            if col in self.encoders and col in X.columns:
                try:
                    X[col] = self.encoders[col].transform(X[col].astype(str))
                except:
                    X[col] = 0

        X = X.fillna(0)

        # Ensure all features exist
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)

        # Predict with both models
        rf_pred = self.rf_model.predict(X_scaled)[0]
        rf_proba = self.rf_model.predict_proba(X_scaled)[0]
        rf_confidence = float(max(rf_proba))

        gb_pred = self.gb_model.predict(X_scaled)[0]
        gb_proba = self.gb_model.predict_proba(X_scaled)[0]
        gb_confidence = float(max(gb_proba))

        # Hybrid prediction
        rf_proba_adj = rf_proba * self.rf_weight
        gb_proba_adj = gb_proba * self.gb_weight
        hybrid_proba = rf_proba_adj + gb_proba_adj
        hybrid_pred_idx = np.argmax(hybrid_proba)
        hybrid_pred = self.rf_model.classes_[hybrid_pred_idx]
        hybrid_confidence = float(max(hybrid_proba))

        # Calculate score
        base_scores = {'Low': 35, 'Medium': 65, 'High': 90}
        base = base_scores.get(hybrid_pred, 50)
        performance_score = base + (hybrid_confidence - 0.5) * 20
        performance_score = round(min(100, max(0, performance_score)), 2)

        # Get next quarter
        quarter_map = {'Q1': 'Q2', 'Q2': 'Q3', 'Q3': 'Q4', 'Q4': 'Q1'}
        year = latest.period_year
        quarter = latest.period_quarter
        next_quarter = quarter_map.get(quarter, 'Q1')
        next_year = year if quarter != 'Q4' else year + 1

        # Save prediction
        prediction = PerformancePrediction(
            employee_id=employee_id,
            period_year=next_year,
            period_quarter=next_quarter,
            predicted_score=performance_score,
            predicted_band=hybrid_pred,
            confidence=hybrid_confidence,
            rf_score=float(max(rf_proba)),
            rf_band=rf_pred,
            rf_confidence=rf_confidence,
            gb_score=float(max(gb_proba)),
            gb_band=gb_pred,
            gb_confidence=gb_confidence,
            algorithm_used="Hybrid",
            feature_importance=json.dumps(dict(zip(self.feature_names, self.rf_model.feature_importances_)))
        )

        self.db.add(prediction)
        self.db.commit()
        self.db.refresh(prediction)

        # Also save to performance_results (legacy)
        result = PerformanceResult(
            employee_id=employee_id,
            performance_score=performance_score,
            performance_band=hybrid_pred,
            confidence=hybrid_confidence,
            model_version="v2.0"
        )
        self.db.add(result)
        self.db.commit()

        return {
            'employee_id': employee_id,
            'predicted_band': hybrid_pred,
            'predicted_score': performance_score,
            'confidence': hybrid_confidence,
            'next_quarter': f"{next_year} {next_quarter}",
            'rf_prediction': rf_pred,
            'rf_confidence': rf_confidence,
            'gb_prediction': gb_pred,
            'gb_confidence': gb_confidence,
            'feature_importance': dict(zip(self.feature_names, self.rf_model.feature_importances_))
        }

    def predict_all_employees(self) -> List[Dict]:
        """Predict performance for all employees with progress tracking"""
        if self.rf_model is None or self.gb_model is None:
            self._load_models()

        if self.rf_model is None or self.gb_model is None:
            raise Exception("Models not trained")

        # Get all employees from history
        employees = self.db.query(PerformanceHistory.employee_id).distinct().all()
        total = len(employees)
        print(f"\n📊 Starting batch prediction for {total} employees...")

        results = []
        success_count = 0
        error_count = 0

        for idx, emp in enumerate(employees, 1):
            try:
                result = self.predict_employee(emp[0])
                if result:
                    results.append(result)
                    success_count += 1
                    if idx % 5 == 0 or idx == total:
                        print(f"  Progress: {idx}/{total} ({success_count} successful)")
            except Exception as e:
                error_count += 1
                print(f"  ❌ Error for {emp[0]}: {e}")

        print(f"\n✅ Batch prediction complete: {success_count} successful, {error_count} errors")
        return results

    def get_employee_predictions(self, employee_id: str) -> List[Dict]:
        """Get prediction history for an employee"""
        predictions = self.db.query(PerformancePrediction).filter_by(
            employee_id=employee_id
        ).order_by(
            PerformancePrediction.period_year,
            PerformancePrediction.period_quarter
        ).all()

        return [{
            'employee_id': p.employee_id,
            'period': f"{p.period_year} {p.period_quarter}",
            'year': p.period_year,
            'quarter': p.period_quarter,
            'predicted_band': p.predicted_band,
            'predicted_score': p.predicted_score,
            'confidence': p.confidence,
            'algorithm': p.algorithm_used,
            'actual_score': p.actual_score,
            'actual_band': p.actual_band,
            'predicted_at': p.predicted_at
        } for p in predictions]

    def get_all_predictions(self) -> List[Dict]:
        """Get all predictions"""
        predictions = self.db.query(PerformancePrediction).order_by(
            PerformancePrediction.predicted_at.desc()
        ).all()

        return [{
            'employee_id': p.employee_id,
            'period': f"{p.period_year} {p.period_quarter}",
            'year': p.period_year,
            'quarter': p.period_quarter,
            'predicted_band': p.predicted_band,
            'predicted_score': p.predicted_score,
            'confidence': p.confidence,
            'algorithm': p.algorithm_used,
            'predicted_at': p.predicted_at
        } for p in predictions]

    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics"""
        predictions = self.db.query(PerformancePrediction).all()

        if not predictions:
            return {
                'total_predictions': 0,
                'band_distribution': {'High': 0, 'Medium': 0, 'Low': 0},
                'average_score': 0,
                'average_confidence': 0,
                'high_performers': 0,
                'medium_performers': 0,
                'low_performers': 0
            }

        band_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        total_score = 0
        total_confidence = 0

        for p in predictions:
            band_counts[p.predicted_band] = band_counts.get(p.predicted_band, 0) + 1
            total_score += p.predicted_score
            total_confidence += p.confidence

        return {
            'total_predictions': len(predictions),
            'band_distribution': band_counts,
            'average_score': total_score / len(predictions),
            'average_confidence': total_confidence / len(predictions),
            'high_performers': band_counts.get('High', 0),
            'medium_performers': band_counts.get('Medium', 0),
            'low_performers': band_counts.get('Low', 0)
        }