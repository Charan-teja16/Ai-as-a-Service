from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Query
from fastapi.responses import FileResponse

from ..schemas import (
    AutoEverythingRequest,
    AutoEverythingResponse,
    AutoSelectRequest,
    AutoSelectResponse,
    DatasetListResponse,
    DatasetSampleImagesResponse,
    ImageUploadResponse,
    DetectTargetRequest,
    DetectTargetResponse,
    ModelCatalogResponse,
    ModelListResponse,
    ModelInfoListResponse,
    ModelInfoEntry,
    ModelDetailResponse,
    PredictRequest,
    PredictResponse,
    TrainAllRequest,
    TrainAllResponse,
    TrainRequest,
    TrainResponse,
    TrainingHistoryResponse,
    UploadResponse,
)
from ..services.report_service import ReportService
from ..services.training_service import TrainingService
from ..services.user_service import UserService
from ..services.email_service import EmailService
from ..utils import analysis
from ..routers.auth import get_current_user
from ..models.user import User
from ..ml.registry import REGISTRY

router = APIRouter()
service = TrainingService()
report_service = ReportService()
user_service = UserService()
email_service = EmailService()


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user)
):
    try:
        payload = service.upload_dataset(file, user.user_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return UploadResponse(**payload)


@router.post("/upload/images/supervised", response_model=ImageUploadResponse)
async def upload_images_supervised(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    try:
        payload = service.upload_image_supervised(file, user.user_id)
        return ImageUploadResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/upload/images/unsupervised", response_model=ImageUploadResponse)
async def upload_images_unsupervised(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    try:
        payload = service.upload_image_unsupervised(file, user.user_id)
        return ImageUploadResponse(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/datasets", response_model=DatasetListResponse)
async def list_datasets(
    mode: Optional[str] = Query(None, description="Filter by mode: csv, supervised, unsupervised"),
    user: User = Depends(get_current_user)
):
    """List all datasets for the current user, optionally filtered by mode."""
    try:
        datasets = service.datasets.list_user_datasets(user.user_id)
        # Filter by mode if provided
        if mode:
            datasets = [d for d in datasets if d.mode == mode]
        # Sort by most recent first (reverse order)
        datasets.reverse()
        # Convert to summary format and remove duplicates by filename
        seen_filenames = set()
        summaries = []
        for ds in datasets:
            # Only add if we haven't seen this filename before
            if ds.filename not in seen_filenames:
                seen_filenames.add(ds.filename)
                summaries.append({
                    "dataset_id": ds.dataset_id,
                    "filename": ds.filename,
                    "mode": ds.mode,
                    "row_count": ds.row_count if ds.mode == "csv" else None,
                    "total_images": ds.total_images if ds.mode in ("supervised", "unsupervised") else None,
                    "columns": ds.columns,
                })
        return DatasetListResponse(datasets=summaries)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/datasets/{dataset_id}/load", response_model=UploadResponse)
async def load_dataset(
    dataset_id: str,
    user: User = Depends(get_current_user)
):
    """Load a dataset by ID (for CSV datasets)."""
    try:
        record = service.datasets.get(dataset_id)
        if record.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        if record.mode != "csv":
            raise HTTPException(status_code=400, detail="This endpoint only supports CSV datasets")
        df = service.datasets.load_dataframe(dataset_id)
        target, confidence, reason, problem_type = analysis.detect_target_column(df)
        preview = df.head(20).fillna("").to_dict(orient="records")
        return UploadResponse(
            dataset_id=record.dataset_id,
            filename=record.filename,
            columns=record.columns,
            row_count=record.row_count,
            preview=preview,
            suggested_target=target,
            target_reason=reason,
            suggested_problem_type=problem_type,
            confidence=confidence,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/datasets/{dataset_id}/load-image", response_model=ImageUploadResponse)
async def load_image_dataset(
    dataset_id: str,
    user: User = Depends(get_current_user)
):
    """Load an image dataset by ID."""
    try:
        record = service.datasets.get(dataset_id)
        if record.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        if record.mode not in ("supervised", "unsupervised"):
            raise HTTPException(status_code=400, detail="This endpoint only supports image datasets")
        return ImageUploadResponse(
            dataset_id=record.dataset_id,
            filename=record.filename,
            mode=record.mode,
            total_images=record.total_images,
            classes=[{"label": c["label"], "count": c["count"]} for c in record.classes],
            message=f"{record.mode.capitalize()} image dataset loaded",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/datasets/{dataset_id}/sample-images", response_model=DatasetSampleImagesResponse)
async def get_sample_images(
    dataset_id: str,
    user: User = Depends(get_current_user)
):
    """Get one sample image from each class in a supervised image dataset."""
    try:
        from ..services.dataset_manager import DatasetManager
        from pathlib import Path
        import base64
        from PIL import Image as PILImage
        
        dataset_mgr = DatasetManager()
        record = dataset_mgr.get(dataset_id)
        
        if record.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        if record.mode != "supervised":
            raise HTTPException(status_code=400, detail="This endpoint only supports supervised image datasets")
        
        dataset_path, _ = dataset_mgr.get_image_paths(dataset_id)
        samples = []
        
        # Get one image from each class
        for class_info in record.classes:
            class_label = class_info["label"]
            class_dir = dataset_path / class_label
            
            if class_dir.exists() and class_dir.is_dir():
                # Find first image in class directory
                ALLOWED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
                image_files = []
                for ext in ALLOWED_IMAGE_EXTS:
                    image_files.extend(list(class_dir.glob(f"*{ext}")))
                    image_files.extend(list(class_dir.glob(f"*{ext.upper()}")))
                
                if image_files:
                    img_path = image_files[0]
                    try:
                        # Read and encode image as base64
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                            samples.append({
                                "class_label": class_label,
                                "image_b64": img_b64
                            })
                    except Exception:
                        continue
        
        return DatasetSampleImagesResponse(samples=samples)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/detect-target", response_model=DetectTargetResponse)
async def detect_target(
    request: DetectTargetRequest,
    user: User = Depends(get_current_user)
):
    try:
        return service.detect_target(request.dataset_id, request.preferred_target, user.user_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/auto-select", response_model=AutoSelectResponse)
async def auto_select(
    request: AutoSelectRequest,
    user: User = Depends(get_current_user)
):
    try:
        # Check usage limits
        if not user.is_subscribed and user.free_runs_remaining <= 0:
            raise HTTPException(
                status_code=403,
                detail="No free runs remaining. Please subscribe to continue."
            )
        result = service.auto_select(request.dataset_id, request.target_column, user.user_id)
        # Only decrement free runs if auto-select succeeded and has valid results
        if result and result.leaderboard and len(result.leaderboard) > 0 and not user.is_subscribed:
            user_service.decrement_free_runs(user.user_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/auto-everything", response_model=AutoEverythingResponse)
async def auto_everything(
    request: AutoEverythingRequest,
    user: User = Depends(get_current_user)
):
    try:
        # Check usage limits
        if not user.is_subscribed and user.free_runs_remaining <= 0:
            raise HTTPException(
                status_code=403,
                detail="No free runs remaining. Please subscribe to continue."
            )
        result = service.auto_everything(request.dataset_id, user.user_id)
        # Only decrement free runs if auto-everything succeeded and has valid results
        if result and result.leaderboard and len(result.leaderboard) > 0 and not user.is_subscribed:
            user_service.decrement_free_runs(user.user_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/train", response_model=TrainResponse)
async def train(
    request: TrainRequest,
    user: User = Depends(get_current_user)
):
    try:
        # Check usage limits
        if not user.is_subscribed and user.free_runs_remaining <= 0:
            raise HTTPException(
                status_code=403,
                detail="No free runs remaining. Please subscribe to continue."
            )
        # Get intensity from request if available, default to medium
        intensity = getattr(request, "intensity", "medium")
        result = service.train_model(
            dataset_id=request.dataset_id,
            target_column=request.target_column,
            model_key=request.model_key,
            problem_type=request.problem_type,
            user_id=user.user_id,
            intensity=intensity,
        )
        # Only decrement free runs if training succeeded and result is valid
        if result and result.model and not user.is_subscribed:
            user_service.decrement_free_runs(user.user_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/train-all", response_model=TrainAllResponse)
async def train_all(
    request: TrainAllRequest,
    user: User = Depends(get_current_user)
):
    try:
        # Check usage limits
        if not user.is_subscribed and user.free_runs_remaining <= 0:
            raise HTTPException(
                status_code=403,
                detail="No free runs remaining. Please subscribe to continue."
            )
        # Get intensity from request if available, default to medium
        intensity = getattr(request, "intensity", "medium")
        result = service.train_all(
            dataset_id=request.dataset_id,
            target_column=request.target_column,
            problem_type=request.problem_type,
            user_id=user.user_id,
            intensity=intensity,
        )
        # Only decrement free runs if training succeeded and leaderboard has results
        if result and result.leaderboard and len(result.leaderboard) > 0 and not user.is_subscribed:
            user_service.decrement_free_runs(user.user_id)
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: PredictRequest,
    user: User = Depends(get_current_user)
):
    try:
        # Verify user owns this model
        stored = service.models.get(request.model_id)
        if stored.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        return service.predict(request.model_id, request.records)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/models", response_model=ModelListResponse)
async def list_models(user: User = Depends(get_current_user)):
    return service.list_models(user.user_id)


@router.get("/models/catalog", response_model=ModelCatalogResponse)
async def list_catalog(dataset_mode: Optional[str] = Query(None, description="Filter by dataset mode: csv, supervised, unsupervised")):
    """Get available models, optionally filtered by dataset mode (csv, supervised, unsupervised)."""
    return service.catalog(dataset_mode=dataset_mode)

@router.get("/models/download/{model_id}")
async def download_model(
    model_id: str,
    user: User = Depends(get_current_user)
):
    try:
        stored = service.models.get(model_id)
        # Verify user owns this model
        if stored.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(stored.path, filename=f"{stored.model_name}.pkl")


@router.get("/models/download/{model_id}/h5")
async def download_model_h5(
    model_id: str,
    user: User = Depends(get_current_user)
):
    try:
        stored = service.models.get(model_id)
        # Verify user owns this model
        if stored.user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return FileResponse(stored.h5_path, filename=f"{stored.model_name}.h5")


@router.get("/report/{report_id}")
async def get_report_info(
    report_id: str,
    user: User = Depends(get_current_user)
):
    """Get report information for preview."""
    try:
        record = report_service.get(report_id)
        # Verify user owns this report (check context)
        # Handle both old reports (without user_id) and new reports (with user_id)
        report_user_id = record.context.get("user_id")
        if report_user_id is not None and report_user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Format leaderboard data for frontend
        leaderboard = record.context.get("leaderboard", [])
        formatted_leaderboard = []
        for item in leaderboard:
            if isinstance(item, dict):
                formatted_leaderboard.append({
                    "model_name": item.get("model_name", "Unknown"),
                    "model_key": item.get("model_key", "unknown"),
                    "rank": item.get("rank", 0),
                    "metrics": item.get("metrics", {}),
                })
        
        return {
            "report_id": report_id,
            "target_column": record.context.get("target_column", "unknown"),
            "problem_type": record.context.get("problem_type", "unknown"),
            "leaderboard": formatted_leaderboard,
            "created_at": record.created_at,
            # Extra context so frontend can restore workspace + dataset
            "dataset_id": record.context.get("dataset_id"),
            "dataset_mode": record.context.get("dataset_mode"),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Report not found: {str(exc)}")


@router.get("/report/download/{report_id}")
async def download_report(
    report_id: str,
    user: User = Depends(get_current_user)
):
    try:
        record = report_service.get(report_id)
        # Verify user owns this report (check context)
        # Handle both old reports (without user_id) and new reports (with user_id)
        report_user_id = record.context.get("user_id")
        if report_user_id is not None and report_user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        import os
        from pathlib import Path
        
        # Handle both absolute and relative paths
        report_path_str = record.path
        if os.path.isabs(report_path_str):
            # Already absolute path
            report_path = Path(report_path_str)
        else:
            # Relative path - join with current working directory
            report_path = Path(os.path.join(os.getcwd(), report_path_str.lstrip("/")))
        
        # Resolve to absolute path
        report_path = report_path.resolve()
        
        if not report_path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Report file not found at {report_path}. The report may not have been generated yet."
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Report not found: {str(exc)}")
    # Use resolved absolute path for FileResponse
    return FileResponse(
        str(report_path),
        filename=f"report-{report_id}.pdf",
        media_type="application/pdf"
    )


@router.post("/reports/{report_id}/send-email")
async def send_report_email(
    report_id: str,
    user: User = Depends(get_current_user)
):
    """Send report to user's email."""
    try:
        record = report_service.get(report_id)
        # Verify user owns this report
        # Handle both old reports (without user_id) and new reports (with user_id)
        report_user_id = record.context.get("user_id")
        if report_user_id is not None and report_user_id != user.user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        import os
        from pathlib import Path
        
        # Handle both absolute and relative paths
        report_path_str = record.path
        if os.path.isabs(report_path_str):
            # Already absolute path
            report_path = Path(report_path_str)
        else:
            # Relative path - join with current working directory
            report_path = Path(os.path.join(os.getcwd(), report_path_str.lstrip("/")))
        
        # Resolve to absolute path
        report_path = report_path.resolve()
        
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")
        
        success = email_service.send_report(user.email, report_path, user.username)
        if success:
            return {"message": "Report sent to your email successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/training-history", response_model=TrainingHistoryResponse)
async def get_training_history(
    user: User = Depends(get_current_user)
):
    """Get training history for current user."""
    history = service.history.get_user_history(user.user_id)
    # Ensure dataset_mode is always included in response
    history_dicts = []
    for h in history:
        h_dict = h.__dict__.copy()
        # Ensure dataset_mode is always present (default to "csv" for backward compatibility)
        if "dataset_mode" not in h_dict or not h_dict["dataset_mode"]:
            h_dict["dataset_mode"] = "csv"
        history_dicts.append(h_dict)
    return TrainingHistoryResponse(history=history_dicts)


# Model Information Database
MODEL_INFO_DATABASE = {
    "linear_regression": {
        "description": "A simple yet powerful algorithm that finds the best-fit line through data points. Perfect for predicting continuous values like prices, temperatures, or sales.",
        "advantages": [
            "Fast training and prediction",
            "Highly interpretable - easy to understand coefficients",
            "No hyperparameter tuning needed",
            "Works well with linear relationships"
        ],
        "disadvantages": [
            "Assumes linear relationship between features and target",
            "Sensitive to outliers",
            "Cannot capture non-linear patterns",
            "Requires feature scaling for best results"
        ],
        "use_cases": [
            "Price prediction (houses, stocks)",
            "Sales forecasting",
            "Temperature prediction",
            "Risk assessment"
        ],
        "complexity": "Low",
        "training_speed": "Fast",
        "interpretability": "High",
        "how_it_works": "Linear Regression finds the best straight line (y = mx + b) that minimizes the distance between predicted and actual values. It uses the least squares method to calculate the optimal slope and intercept.",
        "algorithm_steps": [
            "Calculate the mean of X and Y",
            "Compute the slope (m) using covariance and variance",
            "Calculate the intercept (b) using the means",
            "Form the equation: y = mx + b",
            "Use the equation to make predictions"
        ],
        "hyperparameters": {
            "fit_intercept": "Whether to calculate the intercept (usually True)",
            "normalize": "Whether to normalize features (deprecated, use preprocessing)"
        },
        "visualization_type": "regression_line"
    },
    "logistic_regression": {
        "description": "A classification algorithm that uses a sigmoid function to predict probabilities. Great for binary classification problems like spam detection or medical diagnosis.",
        "advantages": [
            "Fast and efficient",
            "Provides probability scores, not just predictions",
            "Less prone to overfitting than complex models",
            "Works well with small datasets"
        ],
        "disadvantages": [
            "Assumes linear relationship between features and log-odds",
            "Requires feature scaling",
            "May struggle with non-linear boundaries",
            "Sensitive to correlated features"
        ],
        "use_cases": [
            "Email spam detection",
            "Medical diagnosis (disease/no disease)",
            "Credit approval",
            "Customer churn prediction"
        ],
        "complexity": "Low",
        "training_speed": "Fast",
        "interpretability": "High",
        "how_it_works": "Logistic Regression applies a sigmoid (S-shaped) function to linear combinations of features. This transforms the output into probabilities between 0 and 1, making it perfect for classification.",
        "algorithm_steps": [
            "Start with linear combination: z = w₁x₁ + w₂x₂ + ... + b",
            "Apply sigmoid function: σ(z) = 1/(1 + e⁻ᶻ)",
            "Output is probability between 0 and 1",
            "Classify as 1 if probability > 0.5, else 0",
            "Optimize weights using gradient descent"
        ],
        "hyperparameters": {
            "C": "Inverse of regularization strength (higher = less regularization)",
            "penalty": "Regularization type (l1, l2, or none)",
            "solver": "Algorithm for optimization (liblinear, lbfgs, etc.)"
        },
        "visualization_type": "decision_boundary"
    },
    "naive_bayes": {
        "description": "A probabilistic classifier based on Bayes' theorem with a 'naive' assumption of feature independence. Excellent for text classification and spam filtering.",
        "advantages": [
            "Very fast training and prediction",
            "Works well with small datasets",
            "Handles multiple classes naturally",
            "Good for text classification"
        ],
        "disadvantages": [
            "Assumes features are independent (often not true)",
            "Can be outperformed by more complex models",
            "Requires smoothing for unseen features",
            "Sensitive to irrelevant features"
        ],
        "use_cases": [
            "Text classification (spam, sentiment)",
            "Document categorization",
            "Medical diagnosis",
            "Weather prediction"
        ],
        "complexity": "Low",
        "training_speed": "Fast",
        "interpretability": "Medium",
        "how_it_works": "Naive Bayes calculates the probability of each class given the features, using Bayes' theorem. It assumes features are independent, which simplifies calculations but may not always reflect reality.",
        "algorithm_steps": [
            "Calculate prior probability of each class P(class)",
            "For each feature, calculate likelihood P(feature|class)",
            "Apply Bayes' theorem: P(class|features) ∝ P(class) × Π P(feature|class)",
            "Select class with highest probability",
            "Use Laplace smoothing to handle zero probabilities"
        ],
        "hyperparameters": {
            "alpha": "Smoothing parameter (prevents zero probabilities)",
            "fit_prior": "Whether to learn class prior probabilities"
        },
        "visualization_type": "probability_heatmap"
    },
    "decision_tree": {
        "description": "A tree-like model that makes decisions by asking a series of yes/no questions. Each branch represents a decision, leading to a final prediction. Highly interpretable!",
        "advantages": [
            "Very easy to understand and visualize",
            "No feature scaling needed",
            "Handles non-linear relationships",
            "Can work with mixed data types"
        ],
        "disadvantages": [
            "Prone to overfitting",
            "Unstable (small data changes = different tree)",
            "Can create biased trees if classes are imbalanced",
            "May not generalize well"
        ],
        "use_cases": [
            "Medical diagnosis",
            "Credit risk assessment",
            "Customer segmentation",
            "Feature importance analysis"
        ],
        "complexity": "Medium",
        "training_speed": "Fast",
        "interpretability": "High",
        "how_it_works": "Decision Trees split data recursively by asking questions about features. At each node, it chooses the feature that best separates the classes, creating branches until reaching leaf nodes with predictions.",
        "algorithm_steps": [
            "Start with all data at root node",
            "Find best feature to split on (using Gini/entropy)",
            "Create branches for each split value",
            "Recursively split each branch",
            "Stop when stopping criteria met (max depth, min samples)",
            "Assign class/label to leaf nodes"
        ],
        "hyperparameters": {
            "max_depth": "Maximum depth of the tree",
            "min_samples_split": "Minimum samples required to split a node",
            "min_samples_leaf": "Minimum samples required in a leaf",
            "criterion": "Function to measure split quality (gini, entropy)"
        },
        "visualization_type": "tree"
    },
    "random_forest": {
        "description": "An ensemble method that combines multiple decision trees. Each tree votes on the prediction, and the majority wins. More robust and accurate than a single tree!",
        "advantages": [
            "Reduces overfitting compared to single trees",
            "Handles missing values well",
            "Provides feature importance scores",
            "Works well out-of-the-box"
        ],
        "disadvantages": [
            "Less interpretable than single trees",
            "Can be slow with many trees",
            "Requires more memory",
            "May overfit on noisy data"
        ],
        "use_cases": [
            "Feature importance ranking",
            "Medical diagnosis",
            "Stock market prediction",
            "Customer behavior analysis"
        ],
        "complexity": "Medium",
        "training_speed": "Medium",
        "interpretability": "Medium",
        "how_it_works": "Random Forest creates many decision trees, each trained on a random subset of data and features. During prediction, all trees vote, and the most common prediction wins. This 'wisdom of crowds' approach reduces errors.",
        "algorithm_steps": [
            "Create multiple bootstrap samples from training data",
            "For each sample, train a decision tree on random feature subset",
            "Repeat to create a 'forest' of trees",
            "For prediction, get prediction from each tree",
            "Return the majority vote (classification) or average (regression)"
        ],
        "hyperparameters": {
            "n_estimators": "Number of trees in the forest",
            "max_depth": "Maximum depth of each tree",
            "min_samples_split": "Minimum samples to split a node",
            "max_features": "Number of features to consider for each split"
        },
        "visualization_type": "feature_importance"
    },
    "svm": {
        "description": "Support Vector Machine finds the optimal boundary (hyperplane) that maximizes the margin between classes. Powerful for both linear and non-linear classification.",
        "advantages": [
            "Effective in high-dimensional spaces",
            "Memory efficient (uses support vectors only)",
            "Versatile (can use different kernels)",
            "Works well with clear margin of separation"
        ],
        "disadvantages": [
            "Doesn't perform well on large datasets",
            "Sensitive to feature scaling",
            "Doesn't provide probability estimates directly",
            "Can be slow to train"
        ],
        "use_cases": [
            "Text classification",
            "Image recognition",
            "Face detection",
            "Handwriting recognition"
        ],
        "complexity": "High",
        "training_speed": "Slow",
        "interpretability": "Low",
        "how_it_works": "SVM finds the optimal hyperplane that separates classes with maximum margin. Support vectors are the data points closest to the boundary. The algorithm maximizes the distance between the boundary and these support vectors.",
        "algorithm_steps": [
            "Map data to higher-dimensional space (if using kernel)",
            "Find support vectors (points closest to boundary)",
            "Calculate optimal hyperplane with maximum margin",
            "Use support vectors to define decision boundary",
            "Classify new points based on which side of boundary"
        ],
        "hyperparameters": {
            "C": "Regularization parameter (higher = harder margin)",
            "kernel": "Kernel type (linear, rbf, poly, sigmoid)",
            "gamma": "Kernel coefficient (for rbf, poly, sigmoid)"
        },
        "visualization_type": "decision_boundary"
    },
    "knn": {
        "description": "K-Nearest Neighbors is a simple, instance-based learning algorithm. It classifies by finding the K most similar training examples and voting on their labels.",
        "advantages": [
            "Simple and intuitive",
            "No training phase (lazy learning)",
            "Adapts easily to new data",
            "Works well for non-linear problems"
        ],
        "disadvantages": [
            "Slow prediction (must compute distances to all points)",
            "Sensitive to irrelevant features",
            "Requires feature scaling",
            "Memory intensive (stores all training data)"
        ],
        "use_cases": [
            "Recommendation systems",
            "Pattern recognition",
            "Image classification",
            "Anomaly detection"
        ],
        "complexity": "Low",
        "training_speed": "N/A (lazy learning)",
        "interpretability": "Medium",
        "how_it_works": "KNN stores all training examples. For a new prediction, it finds the K nearest neighbors (using distance metrics like Euclidean), then predicts based on the majority class (classification) or average value (regression) of these neighbors.",
        "algorithm_steps": [
            "Store all training examples",
            "For new data point, calculate distance to all training points",
            "Select K nearest neighbors",
            "For classification: majority vote of neighbors' classes",
            "For regression: average of neighbors' values"
        ],
        "hyperparameters": {
            "n_neighbors": "Number of neighbors to consider (K)",
            "weights": "Weight function (uniform or distance-based)",
            "metric": "Distance metric (euclidean, manhattan, etc.)"
        },
        "visualization_type": "knn"
    },
    "gradient_boosting": {
        "description": "An ensemble method that builds models sequentially, where each new model corrects the errors of previous ones. Creates a strong learner from many weak learners.",
        "advantages": [
            "High predictive accuracy",
            "Handles mixed data types",
            "Provides feature importance",
            "Works well on structured data"
        ],
        "disadvantages": [
            "Can overfit if not tuned properly",
            "Training can be slow",
            "Less interpretable than simpler models",
            "Requires careful hyperparameter tuning"
        ],
        "use_cases": [
            "Competition-winning solutions",
            "Fraud detection",
            "Click-through rate prediction",
            "Ranking problems"
        ],
        "complexity": "High",
        "training_speed": "Slow",
        "interpretability": "Low",
        "how_it_works": "Gradient Boosting builds models sequentially. Each new model is trained to predict the residual errors of the previous models. The final prediction is the sum of all models' predictions, gradually improving accuracy.",
        "algorithm_steps": [
            "Start with initial model (usually mean/median)",
            "Calculate residuals (errors) of current model",
            "Train new model to predict these residuals",
            "Add new model to ensemble with small learning rate",
            "Repeat until stopping criteria met",
            "Final prediction = sum of all models"
        ],
        "hyperparameters": {
            "n_estimators": "Number of boosting stages",
            "learning_rate": "Step size shrinkage",
            "max_depth": "Maximum depth of individual trees",
            "subsample": "Fraction of samples used for each tree"
        },
        "visualization_type": "feature_importance"
    },
    "xgboost": {
        "description": "Extreme Gradient Boosting is an optimized implementation of gradient boosting. It's faster, more efficient, and often more accurate than standard gradient boosting.",
        "advantages": [
            "Very fast and efficient",
            "Excellent predictive performance",
            "Built-in regularization",
            "Handles missing values automatically"
        ],
        "disadvantages": [
            "Requires hyperparameter tuning",
            "Less interpretable",
            "Can overfit with small datasets",
            "More complex than simpler models"
        ],
        "use_cases": [
            "Kaggle competitions",
            "Large-scale machine learning",
            "Real-time predictions",
            "Feature engineering validation"
        ],
        "complexity": "High",
        "training_speed": "Medium",
        "interpretability": "Low",
        "how_it_works": "XGBoost is an advanced gradient boosting algorithm with optimizations like parallel tree construction, regularization, and efficient handling of sparse data. It builds trees level-by-level rather than depth-by-depth.",
        "algorithm_steps": [
            "Build trees level-by-level (not depth-by-depth)",
            "Use parallel processing for speed",
            "Apply L1 and L2 regularization",
            "Handle missing values automatically",
            "Use second-order gradient information",
            "Prune trees to prevent overfitting"
        ],
        "hyperparameters": {
            "n_estimators": "Number of boosting rounds",
            "max_depth": "Maximum tree depth",
            "learning_rate": "Step size shrinkage (eta)",
            "subsample": "Row sampling ratio",
            "colsample_bytree": "Column sampling ratio"
        },
        "visualization_type": "feature_importance"
    },
    "image_cnn": {
        "description": "Convolutional Neural Network is a deep learning architecture designed for image recognition. It uses convolutional layers to automatically learn spatial features like edges, shapes, and patterns.",
        "advantages": [
            "Automatic feature extraction",
            "Excellent for image data",
            "Handles translation, rotation, scaling",
            "State-of-the-art image classification"
        ],
        "disadvantages": [
            "Requires large amounts of data",
            "Computationally expensive",
            "Black box (hard to interpret)",
            "Needs GPU for fast training"
        ],
        "use_cases": [
            "Image classification",
            "Object detection",
            "Medical image analysis",
            "Face recognition"
        ],
        "complexity": "High",
        "training_speed": "Slow",
        "interpretability": "Low",
        "how_it_works": "CNNs use convolutional layers to scan images with filters (kernels) that detect features like edges and textures. Pooling layers reduce dimensionality, and fully connected layers make final predictions. The network learns hierarchical features automatically.",
        "algorithm_steps": [
            "Apply convolutional filters to detect low-level features (edges)",
            "Use activation functions (ReLU) to introduce non-linearity",
            "Apply pooling to reduce spatial dimensions",
            "Stack multiple conv-pool blocks for hierarchical features",
            "Flatten and pass through dense layers",
            "Output class probabilities using softmax"
        ],
        "hyperparameters": {
            "filters": "Number of convolutional filters per layer",
            "kernel_size": "Size of convolutional kernels",
            "pool_size": "Size of pooling windows",
            "dense_units": "Number of neurons in dense layers",
            "dropout_rate": "Fraction of neurons to randomly disable",
            "learning_rate": "Step size for weight updates"
        },
        "visualization_type": "cnn_architecture"
    },
    "image_kmeans": {
        "description": "K-Means Clustering groups similar images together without labels. It finds K clusters by minimizing the distance between points and their cluster centers.",
        "advantages": [
            "Simple and fast",
            "Works without labels (unsupervised)",
            "Easy to understand",
            "Good for exploratory data analysis"
        ],
        "disadvantages": [
            "Requires specifying number of clusters (K)",
            "Sensitive to initialization",
            "Assumes spherical clusters",
            "May converge to local optima"
        ],
        "use_cases": [
            "Image segmentation",
            "Customer segmentation",
            "Document clustering",
            "Color quantization"
        ],
        "complexity": "Medium",
        "training_speed": "Fast",
        "interpretability": "Medium",
        "how_it_works": "K-Means randomly initializes K cluster centers, then iteratively assigns each data point to the nearest center and updates centers to the mean of their assigned points. This process repeats until convergence.",
        "algorithm_steps": [
            "Randomly initialize K cluster centers",
            "Assign each point to nearest center",
            "Update each center to mean of its points",
            "Repeat assignment and update steps",
            "Stop when centers no longer change",
            "Return cluster assignments"
        ],
        "hyperparameters": {
            "n_clusters": "Number of clusters to form (K)",
            "init": "Initialization method (k-means++, random)",
            "max_iter": "Maximum iterations",
            "n_init": "Number of times to run with different seeds"
        },
        "visualization_type": "clusters"
    }
}


@router.get("/models/info", response_model=ModelInfoListResponse)
async def list_models_info(
    problem_type: Optional[str] = Query(None, description="Filter by problem type"),
    dataset_mode: Optional[str] = Query(None, description="Filter by dataset mode"),
    user: User = Depends(get_current_user)
):
    """List all available models with their information."""
    try:
        models = REGISTRY.list_models(problem_type=problem_type, dataset_mode=dataset_mode)
        model_entries = []
        for model in models:
            model_key = model["key"]
            if model_key in MODEL_INFO_DATABASE:
                info = MODEL_INFO_DATABASE[model_key]
                model_entries.append(ModelInfoEntry(
                    key=model_key,
                    name=model["name"],
                    description=info["description"],
                    problem_types=model["problem_types"],
                    dataset_modes=model["dataset_modes"],
                    advantages=info["advantages"],
                    disadvantages=info["disadvantages"],
                    use_cases=info["use_cases"],
                    complexity=info["complexity"],
                    training_speed=info["training_speed"],
                    interpretability=info["interpretability"]
                ))
        return ModelInfoListResponse(models=model_entries)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/models/info/{model_key}", response_model=ModelDetailResponse)
async def get_model_info(
    model_key: str,
    user: User = Depends(get_current_user)
):
    """Get detailed information about a specific model."""
    try:
        if model_key not in REGISTRY._registry:
            raise HTTPException(status_code=404, detail=f"Model {model_key} not found")
        
        model_meta = REGISTRY._registry[model_key]
        if model_key not in MODEL_INFO_DATABASE:
            raise HTTPException(status_code=404, detail=f"Model info for {model_key} not available")
        
        info = MODEL_INFO_DATABASE[model_key]
        return ModelDetailResponse(
            key=model_key,
            name=model_meta["name"],
            description=info["description"],
            problem_types=model_meta["problem_types"],
            dataset_modes=model_meta["dataset_modes"],
            advantages=info["advantages"],
            disadvantages=info["disadvantages"],
            use_cases=info["use_cases"],
            complexity=info["complexity"],
            training_speed=info["training_speed"],
            interpretability=info["interpretability"],
            how_it_works=info["how_it_works"],
            algorithm_steps=info["algorithm_steps"],
            hyperparameters=info["hyperparameters"],
            visualization_type=info["visualization_type"]
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

