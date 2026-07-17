import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
from sqlalchemy.orm import Session
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from typing import Optional, List, Dict, Tuple

from backend.models.master import EmployeePerformanceView
from backend.models.performance_results import PerformanceResult
from backend.schemas.performance_schema import PredictionResponse

class PredictionService:
    def __init__(self, db_session: Session):
        self.db = db_session
        self.model = None
        self.encoders = {}
        self.feature_names = []
        self.model_path = Path("models/performance_model.pkl")
        self.encoder_path = Path("models/feature_encoders.pkl")
        self.feature_path = Path("models/feature_names.pkl")

        # Create models directory if it doesn't exist
        self.model_path.parent.mkdir(exist_ok=True)

        # Load model if it exists
        self._load_model()

    def _load_model(self):
        """Load the trained model and encoders"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                print("✅ Model loaded successfully")

            if self.encoder_path.exists():
                self.encoders = joblib.load(self.encoder_path)
                print("✅ Encoders loaded successfully")

            if self.feature_path.exists():
                self.feature_names = joblib.load(self.feature_path)
                print("✅ Feature names loaded successfully")
        except Exception as e:
            print(f"⚠️ Could not load model: {e}")

    def _prepare_data_from_db(self):
        """Fetch data from database and prepare for training"""
        try:
            # Get all data from the view
            records = self.db.query(EmployeePerformanceView).all()

            if not records:
                print("⚠️ No records found in the database")
                return None, None

            # Convert to DataFrame
            data = []
            for r in records:
                data.append({
                    'employee_id': r.employee_id,
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

            # Create performance score based on multiple factors
            df['performance_score'] = (
                    df['deadline_adherence_rate'] * 0.20 +
                    df['completed_storypoint_ratio'] * 100 * 0.20 +
                    df['punctuality'] * 10 * 0.10 +
                    df['problem_solving'] * 10 * 0.10 +
                    df['leadership'] * 10 * 0.10 +
                    df['collaboration'] * 10 * 0.10 +
                    df['communication'] * 10 * 0.10 +
                    df['metric_6_value'] * 10 * 0.10
            )

            # Normalize to 0-100
            df['performance_score'] = df['performance_score'].clip(0, 100)

            # Create performance band
            df['performance_band'] = pd.cut(
                df['performance_score'],
                bins=[-1, 40, 70, 100],
                labels=['Low', 'Medium', 'High']
            )

            return df, df['performance_band']

        except Exception as e:
            print(f"❌ Error preparing data: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def train_model(self) -> Dict:
        """Train the Random Forest model"""
        print("\n🔧 Training Performance Prediction Model...")

        X, y = self._prepare_data_from_db()

        if X is None or y is None:
            raise Exception("No data available for training")

        # Define features to use
        feature_columns = [
            'job_role', 'years_of_experience', 'department',
            'duration_weeks', 'relative_effort', 'team_size',
            'project_complexity', 'rework_count', 'no_pay_leave',
            'blockers', 'punctuality', 'problem_solving',
            'leadership', 'collaboration', 'communication',
            'deadline_adherence_rate', 'avg_response_time',
            'completed_storypoint_ratio', 'completed_story_points',
            'assigned_story_points'
        ]

        X_features = X[feature_columns].copy()

        # Encode categorical variables
        categorical_cols = ['job_role', 'department', 'project_complexity']
        self.encoders = {}

        for col in categorical_cols:
            if col in X_features.columns:
                encoder = LabelEncoder()
                X_features[col] = encoder.fit_transform(X_features[col].astype(str))
                self.encoders[col] = encoder

        # Handle missing values
        X_features = X_features.fillna(X_features.median())

        # Save feature names
        self.feature_names = X_features.columns.tolist()
        joblib.dump(self.feature_names, self.feature_path)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train Random Forest
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')

        print(f"\n📊 Model Performance:")
        print(f"  - Accuracy: {accuracy:.4f}")
        print(f"  - F1 Score: {f1:.4f}")

        # Feature importance
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        print(f"\n🏆 Top 5 Important Features:")
        for feat, imp in top_features:
            print(f"  - {feat}: {imp:.4f}")

        # Save model
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.encoders, self.encoder_path)
        joblib.dump(self.feature_names, self.feature_path)

        print(f"\n✅ Model saved to: {self.model_path}")

        # Return properly formatted top_features
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'top_features': [
                {'feature': feat, 'importance': float(imp)}
                for feat, imp in top_features
            ]
        }

    def predict_employee(self, employee_id: str) -> Optional[PredictionResponse]:
        """Predict performance for a single employee"""
        if self.model is None:
            self._load_model()

        if self.model is None:
            raise Exception("Model not trained. Please train the model first.")

        # Get employee data
        employee_data = self.db.query(EmployeePerformanceView).filter_by(
            employee_id=employee_id
        ).first()

        if not employee_data:
            return None

        # Prepare features
        feature_dict = {
            'job_role': employee_data.job_role or 'Unknown',
            'years_of_experience': employee_data.years_of_experience or 0,
            'department': employee_data.department or 'Unknown',
            'duration_weeks': employee_data.duration_weeks or 0,
            'relative_effort': employee_data.relative_effort or 0,
            'team_size': employee_data.team_size or 1,
            'project_complexity': employee_data.project_complexity or 'Medium',
            'rework_count': employee_data.rework_count or 0,
            'no_pay_leave': employee_data.no_pay_leave or 0,
            'blockers': employee_data.blockers or 0,
            'punctuality': employee_data.punctuality or 3,
            'problem_solving': employee_data.problem_solving or 3,
            'leadership': employee_data.leadership or 3,
            'collaboration': employee_data.collaboration or 3,
            'communication': employee_data.communication or 3,
            'deadline_adherence_rate': employee_data.deadline_adherence_rate or 0,
            'avg_response_time': employee_data.avg_response_time or 0,
            'completed_storypoint_ratio': employee_data.completed_storypoint_ratio or 0,
            'completed_story_points': employee_data.completed_story_points or 0,
            'assigned_story_points': employee_data.assigned_story_points or 0
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

        # Handle missing values
        X = X.fillna(0)

        # Ensure we have all required features
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        # Reorder columns to match training
        X = X[self.feature_names]

        # Make prediction
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        confidence = float(max(probability))

        # Calculate performance score
        performance_score = self._calculate_score(prediction, confidence)

        # Get feature importance for this prediction
        feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]

        # Save to database
        result = PerformanceResult(
            employee_id=employee_id,
            performance_score=performance_score,
            performance_band=prediction,
            confidence=confidence,
            feature_snapshot=json.dumps(top_features),
            model_version='v1.0'
        )

        self.db.add(result)
        self.db.commit()
        self.db.refresh(result)

        return PredictionResponse(
            employee_id=employee_id,
            performance_band=prediction,
            performance_score=performance_score,
            confidence=confidence,
            top_features=top_features,
            predicted_at=result.predicted_at
        )

    def _calculate_score(self, band: str, confidence: float) -> float:
        """Calculate numerical performance score"""
        base_scores = {'Low': 35, 'Medium': 65, 'High': 90}
        base = base_scores.get(band, 50)
        # Adjust based on confidence
        adjustment = (confidence - 0.5) * 20
        return round(min(100, max(0, base + adjustment)), 2)

    def get_employee_history(self, employee_id: str) -> List[Dict]:
        """Get prediction history for an employee"""
        results = self.db.query(PerformanceResult).filter_by(
            employee_id=employee_id
        ).order_by(PerformanceResult.predicted_at.desc()).all()

        return [{
            'employee_id': r.employee_id,
            'performance_band': r.performance_band,
            'performance_score': r.performance_score,
            'confidence': r.confidence,
            'predicted_at': r.predicted_at
        } for r in results]

    def get_all_predictions(self) -> List[Dict]:
        """Get all predictions"""
        results = self.db.query(PerformanceResult).order_by(
            PerformanceResult.predicted_at.desc()
        ).all()

        return [{
            'employee_id': r.employee_id,
            'performance_band': r.performance_band,
            'performance_score': r.performance_score,
            'confidence': r.confidence,
            'predicted_at': r.predicted_at
        } for r in results]

    def get_prediction_stats(self) -> Dict:
        """Get prediction statistics"""
        results = self.db.query(PerformanceResult).all()

        if not results:
            return {
                'total_predictions': 0,
                'band_distribution': {'High': 0, 'Medium': 0, 'Low': 0},
                'average_score': 0,
                'high_performers': 0,
                'medium_performers': 0,
                'low_performers': 0
            }

        # Band distribution
        band_counts = {'High': 0, 'Medium': 0, 'Low': 0}
        for r in results:
            if r.performance_band in band_counts:
                band_counts[r.performance_band] += 1

        # Calculate averages
        scores = [r.performance_score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            'total_predictions': len(results),
            'band_distribution': band_counts,
            'average_score': round(avg_score, 2),
            'high_performers': band_counts['High'],
            'medium_performers': band_counts['Medium'],
            'low_performers': band_counts['Low']
        }