import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import sqlite3
from backend.database.database import engine, SessionLocal
from sqlalchemy import text

def run_migration():
    """Run all database migrations"""

    print("="*60)
    print("🔄 RUNNING DATABASE MIGRATIONS")
    print("="*60)

    conn = sqlite3.connect('data/research_system.db')
    cursor = conn.cursor()

    try:
        # ==========================================
        # STEP 1: Add columns to employee_performance_view
        # ==========================================
        print("\n📝 Step 1: Adding quarter columns to employee_performance_view...")

        # Check if columns already exist
        cursor.execute("PRAGMA table_info(employee_performance_view)")
        columns = [col[1] for col in cursor.fetchall()]

        if 'period_year' not in columns:
            cursor.execute("ALTER TABLE employee_performance_view ADD COLUMN period_year INTEGER")
            print("  ✅ Added period_year column")
        else:
            print("  ⏭️ period_year column already exists")

        if 'period_quarter' not in columns:
            cursor.execute("ALTER TABLE employee_performance_view ADD COLUMN period_quarter TEXT")
            print("  ✅ Added period_quarter column")
        else:
            print("  ⏭️ period_quarter column already exists")

        # ==========================================
        # STEP 2: Create performance_history table
        # ==========================================
        print("\n📝 Step 2: Creating performance_history table...")

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS performance_history (
                                                                          id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                          employee_id TEXT NOT NULL,
                                                                          period_year INTEGER NOT NULL,
                                                                          period_quarter TEXT NOT NULL,
                                                                          institution TEXT,
                                                                          date_of_birth TEXT,
                                                                          gender TEXT,
                                                                          age_group TEXT,
                                                                          job_role TEXT,
                                                                          years_of_experience REAL,
                                                                          department TEXT,
                                                                          language_proficiency TEXT,
                                                                          ethnicity TEXT,
                                                                          educational_institute TEXT,
                                                                          punctuality INTEGER,
                                                                          problem_solving INTEGER,
                                                                          leadership INTEGER,
                                                                          collaboration INTEGER,
                                                                          communication INTEGER,
                                                                          deadline_adherence_rate REAL,
                                                                          adherence_level TEXT,
                                                                          avg_response_time REAL,
                                                                          response_time_level TEXT,
                                                                          no_of_meetings_attended INTEGER,
                                                                          no_of_subordinates INTEGER,
                                                                          decision_contribution INTEGER,
                                                                          learning_hours_per_month REAL,
                                                                          type_of_learning TEXT,
                                                                          team_engagement_frequency INTEGER,
                                                                          completed_storypoint_ratio REAL,
                                                                          completed_story_points REAL,
                                                                          assigned_story_points REAL,
                                                                          project_id TEXT,
                                                                          project_name TEXT,
                                                                          duration_weeks REAL,
                                                                          relative_effort REAL,
                                                                          team_size INTEGER,
                                                                          project_complexity TEXT,
                                                                          rework_count INTEGER,
                                                                          no_pay_leave INTEGER,
                                                                          blockers INTEGER,
                                                                          metric_1_name TEXT,
                                                                          metric_1_value REAL,
                                                                          metric_2_name TEXT,
                                                                          metric_2_value REAL,
                                                                          metric_3_name TEXT,
                                                                          metric_3_value REAL,
                                                                          metric_4_name TEXT,
                                                                          metric_4_value REAL,
                                                                          metric_5_name TEXT,
                                                                          metric_5_value REAL,
                                                                          metric_6_name TEXT,
                                                                          metric_6_value REAL,
                                                                          metric_7_name TEXT,
                                                                          metric_7_value REAL,
                                                                          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                                                          UNIQUE(employee_id, period_year, period_quarter)
                           )
                       """)
        print("  ✅ performance_history table created")

        # ==========================================
        # STEP 3: Create performance_predictions table
        # ==========================================
        print("\n📝 Step 3: Creating performance_predictions table...")

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS performance_predictions (
                                                                              id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                              employee_id TEXT NOT NULL,
                                                                              period_year INTEGER NOT NULL,
                                                                              period_quarter TEXT NOT NULL,
                                                                              predicted_score REAL NOT NULL,
                                                                              predicted_band TEXT NOT NULL,
                                                                              confidence REAL NOT NULL,
                                                                              rf_score REAL,
                                                                              rf_band TEXT,
                                                                              rf_confidence REAL,
                                                                              gb_score REAL,
                                                                              gb_band TEXT,
                                                                              gb_confidence REAL,
                                                                              algorithm_used TEXT NOT NULL,
                                                                              model_version TEXT,
                                                                              feature_importance TEXT,
                                                                              actual_score REAL,
                                                                              actual_band TEXT,
                                                                              predicted_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                                                              evaluated_at DATETIME
                       )
                       """)
        print("  ✅ performance_predictions table created")

        # ==========================================
        # STEP 4: Create model_metadata table
        # ==========================================
        print("\n📝 Step 4: Creating model_metadata table...")

        cursor.execute("""
                       CREATE TABLE IF NOT EXISTS model_metadata (
                                                                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                                                                     model_name TEXT NOT NULL,
                                                                     algorithm_type TEXT NOT NULL,
                                                                     training_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                                                                     accuracy REAL,
                                                                     f1_score REAL,
                                                                     roc_auc REAL,
                                                                     cv_mean REAL,
                                                                     feature_importance TEXT,
                                                                     feature_names TEXT,
                                                                     hyperparameters TEXT,
                                                                     is_active INTEGER DEFAULT 1
                       )
                       """)
        print("  ✅ model_metadata table created")

        # ==========================================
        # STEP 5: Migrate data from employee_performance_view to performance_history
        # ==========================================
        print("\n📝 Step 5: Migrating data to performance_history...")

        # Check if there's data in employee_performance_view
        cursor.execute("SELECT COUNT(*) FROM employee_performance_view")
        count = cursor.fetchone()[0]

        if count > 0:
            # Check if performance_history already has data
            cursor.execute("SELECT COUNT(*) FROM performance_history")
            hist_count = cursor.fetchone()[0]

            if hist_count == 0:
                # Get all columns from employee_performance_view
                cursor.execute("PRAGMA table_info(employee_performance_view)")
                columns = [col[1] for col in cursor.fetchall()]

                # Build column list
                cols_str = ', '.join([f'"{col}"' for col in columns if col not in ['period_year', 'period_quarter']])

                # Insert data with default quarters (Q1 2025 if not specified)
                cursor.execute(f"""
                    INSERT INTO performance_history (
                        employee_id, institution, date_of_birth, gender, age_group,
                        job_role, years_of_experience, department, language_proficiency,
                        ethnicity, educational_institute, punctuality, problem_solving,
                        leadership, collaboration, communication, deadline_adherence_rate,
                        adherence_level, avg_response_time, response_time_level,
                        no_of_meetings_attended, no_of_subordinates, decision_contribution,
                        learning_hours_per_month, type_of_learning, team_engagement_frequency,
                        completed_storypoint_ratio, completed_story_points, assigned_story_points,
                        project_id, project_name, duration_weeks, relative_effort, team_size,
                        project_complexity, rework_count, no_pay_leave, blockers,
                        metric_1_name, metric_1_value, metric_2_name, metric_2_value,
                        metric_3_name, metric_3_value, metric_4_name, metric_4_value,
                        metric_5_name, metric_5_value, metric_6_name, metric_6_value,
                        metric_7_name, metric_7_value, period_year, period_quarter
                    )
                    SELECT 
                        {cols_str},
                        COALESCE(period_year, 2025) as period_year,
                        COALESCE(period_quarter, 'Q1') as period_quarter
                    FROM employee_performance_view
                """)
                print(f"  ✅ Migrated {count} records to performance_history")
            else:
                print(f"  ⏭️ performance_history already has {hist_count} records, skipping migration")
        else:
            print("  ⏭️ No data in employee_performance_view to migrate")

        # ==========================================
        # STEP 6: Create indexes for performance
        # ==========================================
        print("\n📝 Step 6: Creating indexes...")

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_history_employee ON performance_history(employee_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_history_period ON performance_history(period_year, period_quarter)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_employee ON performance_predictions(employee_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_period ON performance_predictions(period_year, period_quarter)")

        print("  ✅ Indexes created")

        # ==========================================
        # STEP 7: Commit changes
        # ==========================================
        conn.commit()

        print("\n" + "="*60)
        print("✅ MIGRATION COMPLETED SUCCESSFULLY!")
        print("="*60)

        # Show summary
        print("\n📊 Database Summary:")
        cursor.execute("SELECT COUNT(*) FROM employee_performance_view")
        print(f"  - employee_performance_view: {cursor.fetchone()[0]} records")

        cursor.execute("SELECT COUNT(*) FROM performance_history")
        print(f"  - performance_history: {cursor.fetchone()[0]} records")

        cursor.execute("SELECT COUNT(*) FROM performance_predictions")
        print(f"  - performance_predictions: {cursor.fetchone()[0]} records")

        cursor.execute("SELECT COUNT(*) FROM model_metadata")
        print(f"  - model_metadata: {cursor.fetchone()[0]} records")

    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    run_migration()