import logging
import joblib
import pandas as pd
from django.db import transaction
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

required_columns = [
    'age',
    'is_dormant',
    'no_products',
    'account_age_months',
    'monthly_deposit',
    'complaints_count',
    'days_since_last_complaint',
    'mobile_banking_active',
    'internet_banking_active',
    'ussd_banking_active',
    'account_linkage_banking_active'
]


def handle_missing_values(df):
    """Handle missing values in the DataFrame."""
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def handle_outliers(df, numerical_cols):
    """Cap outliers using IQR method."""
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    return df


def train_churn_model_from_file(file_path):
    """Training a churn prediction model from a CSV file."""
    try:
        # Read and validate data
        df = pd.read_csv(file_path)
        missing_cols = [col for col in required_columns + ['churned'] if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        logger.info(f"Input DataFrame shape: {df.shape}")
        logger.info(f"Churn distribution: {df['churned'].value_counts().to_dict()}")

        # Handle missing values and outliers
        df = handle_missing_values(df)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'churned' in numerical_cols:
            numerical_cols.remove('churned')
        df = handle_outliers(df, numerical_cols)

        # Identify categorical columns
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()
        if 'churned' in categorical_cols:
            categorical_cols.remove('churned')

        # Scale numerical features
        scaler = StandardScaler()
        if numerical_cols:
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            logger.info("Numerical features scaled successfully")

        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Features and label
        X = df_encoded.drop(columns=['churned'])
        y = df_encoded['churned']

        # Log feature statistics
        logger.info(f"Feature matrix shape: {X.shape}, Target vector shape: {y.shape}")
        logger.info(f"Class balance: {y.value_counts(normalize=True).to_dict()}")

        # Handle class imbalance with SMOTE
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)
        logger.info(
            f"After SMOTE - Feature matrix shape: {X.shape}, Class balance: {pd.Series(y).value_counts(normalize=True).to_dict()}")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

        # Train the model
        base_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5, min_samples_leaf=2,
            random_state=42
        )
        model = CalibratedClassifierCV(base_model, method='sigmoid', cv=5)
        model.fit(X_train, y_train)
        logger.info("Model training completed")

        # Feature importance
        try:
            feature_importances = None
            for clf in model.calibrated_classifiers_:
                if hasattr(clf, 'estimator') and hasattr(clf.estimator, 'feature_importances_'):
                    feature_importances = clf.estimator.feature_importances_
                    break
                elif hasattr(clf, 'base_estimator') and hasattr(clf.base_estimator, 'feature_importances_'):
                    feature_importances = clf.base_estimator.feature_importances_
                    break
            if feature_importances is None:
                logger.warning("Could not access feature importances; using zeros")
                feature_importance = pd.DataFrame({'feature': X.columns, 'importance': [0] * len(X.columns)})
            else:
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': feature_importances
                }).sort_values(by='importance', ascending=False)
            logger.info("Top 5 feature importances:\n" + feature_importance.head().to_string())
        except Exception as e:
            logger.warning(f"Error computing feature importance: {str(e)}; using zeros")
            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': [0] * len(X.columns)})

        # Evaluate on training and test sets
        def evaluate_model(model, X, y, dataset_name):
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]

            # Adjust the metrics to account for 'YES' as positive class
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, pos_label='YES'),
                'recall': recall_score(y, y_pred, pos_label='YES'),
                'f1_score': f1_score(y, y_pred, pos_label='YES'),
                'roc_auc': roc_auc_score(y, y_proba)
            }
            cm = confusion_matrix(y, y_pred)
            logger.info(f"{dataset_name} metrics: {metrics}")
            logger.info(f"{dataset_name} confusion matrix:\n{cm}")
            return metrics

        train_metrics = evaluate_model(model, X_train, y_train, "Training")
        test_metrics = evaluate_model(model, X_test, y_test, "Test")

        # Save artifacts
        joblib.dump(model, 'churn_model.pkl')
        joblib.dump(X.columns.tolist(), 'model_columns.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        logger.info("Model, columns, and scaler saved successfully")

        return test_metrics

    except Exception as e:
        logger.error(f"Error in train_churn_model_from_file: {str(e)}")
        raise


def get_verdict_from_f1(f1_):
    """Return a verdict based on F1 score."""
    if f1_ < 0.3:
        return "❌ Poor"
    elif f1_ < 0.6:
        return "⚠️ Fair"
    else:
        return "✅ Good"


def run_churn_test(df, dj_model):
    """Run churn prediction on a test DataFrame."""
    try:
        # Validate input columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing columns: {missing_cols}")

        logger.info(f"Input DataFrame shape: {df.shape}")

        # Handle missing values and outliers
        df = handle_missing_values(df)
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if 'churned' in numerical_cols:
            numerical_cols.remove('churned')
        df = handle_outliers(df, numerical_cols)

        # Load model artifacts
        model = joblib.load("churn_model.pkl")
        model_columns = joblib.load("model_columns.pkl")
        scaler = joblib.load("scaler.pkl")
        logger.info("Model, columns, and scaler loaded successfully")

        # Identify categorical columns (optional, based on the data)
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns.tolist()

        # Scale numerical features
        if numerical_cols:
            df[numerical_cols] = scaler.transform(df[numerical_cols])
            logger.info("Numerical features scaled successfully")

        # One-hot encode categorical features (if any)
        input_df = df.drop(columns=['churned'], errors='ignore')  # Ensure 'churned' column is dropped
        X = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

        # Align with training columns (from the trained model)
        X = X.reindex(columns=model_columns, fill_value=0)
        logger.info(f"Feature matrix shape after reindexing: {X.shape}")

        # Predict
        predictions = model.predict(X)
        proba = model.predict_proba(X)[:, 1]
        logger.info(f"Mean churn probability: {proba.mean():.3f}")
        logger.info(f"Predicted {len(predictions)} customers, {sum(predictions)} churned")

        # Add predictions to DataFrame
        df['churned_pred'] = predictions
        df['churn_probability'] = proba
        churned_df = df[df['churned_pred'] == 1]

        # Create TestingCustomerData instances
        def make_instance(row):
            try:
                return dj_model(
                    age=int(row['age']),
                    is_dormant=bool(row['is_dormant']),
                    num_products=int(row['no_products']),
                    account_age_months=int(row['account_age_months']),
                    monthly_deposits=float(row['monthly_deposit']),
                    complaints_count=int(row['complaints_count']),
                    days_since_last_complaint=int(row['days_since_last_complaint']),
                    mobile_banking_active=bool(row['mobile_banking_active']),
                    internet_banking_active=bool(row['internet_banking_active']),
                    ussd_banking_active=bool(row['ussd_banking_active']),
                    account_linkage_banking_active=bool(row['account_linkage_banking_active']),
                    churned=bool(row['churned_pred'])
                )
            except Exception as e:
                logger.error(f"Error creating instance: {str(e)}")
                raise

        instances = [make_instance(row) for _, row in churned_df.iterrows()]
        logger.info(f"Created {len(instances)} TestingCustomerData instances")

        # Save to database
        with transaction.atomic():
            dj_model.objects.bulk_create(instances, batch_size=1000, ignore_conflicts=True)
        logger.info(f"Saved {len(instances)} churned customers to database")

        return churned_df

    except Exception as e:
        logger.error(f"Error in run_churn_test: {str(e)}")
        raise

