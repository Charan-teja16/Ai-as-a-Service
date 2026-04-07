from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ProblemType = Literal["classification", "regression", "unsupervised"]


class DatasetPreview(BaseModel):
    dataset_id: str
    filename: str
    columns: List[str]
    row_count: int
    preview: List[Dict[str, Any]]


class UploadResponse(DatasetPreview):
    suggested_target: Optional[str] = None
    target_reason: Optional[str] = None
    suggested_problem_type: Optional[ProblemType] = None
    confidence: Optional[float] = Field(default=None, ge=0, le=1)


class ImageClassSummary(BaseModel):
    label: str
    count: int


class ImageUploadResponse(BaseModel):
    dataset_id: str
    filename: str
    mode: Literal["supervised", "unsupervised"]
    total_images: int
    classes: List[ImageClassSummary] = Field(default_factory=list)
    message: str


class ClassSampleImage(BaseModel):
    class_label: str
    image_b64: str  # Base64 encoded image


class DatasetSampleImagesResponse(BaseModel):
    samples: List[ClassSampleImage]


class DatasetSummary(BaseModel):
    dataset_id: str
    filename: str
    mode: str  # csv | supervised | unsupervised
    row_count: Optional[int] = None
    total_images: Optional[int] = None
    columns: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None


class DatasetListResponse(BaseModel):
    datasets: List[DatasetSummary]


class DetectTargetRequest(BaseModel):
    dataset_id: str
    preferred_target: Optional[str] = None


class DetectTargetResponse(BaseModel):
    dataset_id: str
    target_column: str
    confidence: float = Field(ge=0, le=1)
    problem_type: ProblemType
    reason: str


class TrainRequest(BaseModel):
    dataset_id: str
    target_column: str
    model_key: str
    problem_type: Optional[ProblemType] = None
    intensity: Literal["less", "medium", "rigorous"] = "medium"


class TrainAllRequest(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: Optional[ProblemType] = None
    intensity: Literal["less", "medium", "rigorous"] = "medium"


class AutoSelectRequest(BaseModel):
    dataset_id: str
    target_column: Optional[str] = None


class AutoEverythingRequest(BaseModel):
    dataset_id: str
    accept_target: bool = True
    accept_model: bool = True


class MetricPayload(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    silhouette: Optional[float] = None


class ArtifactPayload(BaseModel):
    confusion_matrix: Optional[List[List[float]]] = None
    residuals: Optional[List[float]] = None
    roc: Optional[Dict[str, List[float]]] = None
    plots: Optional[Dict[str, Optional[str]]] = None


class FeatureHint(BaseModel):
    name: str
    kind: Literal["numeric", "categorical"]
    examples: List[str] = Field(default_factory=list)
    value_map: Optional[Dict[str, int]] = None


class ModelSummary(BaseModel):
    model_id: str
    model_key: str
    model_name: str
    dataset_id: str
    problem_type: ProblemType
    metrics: MetricPayload
    rank: Optional[int] = None
    created_at: datetime
    download_url: Optional[str] = None
    download_h5_url: Optional[str] = None
    report_url: Optional[str] = None
    artifacts: Optional[ArtifactPayload] = None
    columns: List[str] = Field(default_factory=list)
    feature_hints: List[FeatureHint] = Field(default_factory=list)


class ModelCatalogEntry(BaseModel):
    key: str
    name: str
    problem_types: List[ProblemType]
    dataset_modes: List[str] = Field(default_factory=list)


class ModelCatalogResponse(BaseModel):
    models: List[ModelCatalogEntry]


class AutoSelectResponse(BaseModel):
    recommended_model: str
    reason: str
    leaderboard: List[ModelSummary]


class TrainResponse(BaseModel):
    message: str
    model: ModelSummary
    artifacts: ArtifactPayload


class TrainAllResponse(BaseModel):
    message: str
    leaderboard: List[ModelSummary]
    report_id: Optional[str] = None


class AutoEverythingResponse(BaseModel):
    dataset_id: str
    target_column: str
    problem_type: ProblemType
    leaderboard: List[ModelSummary]
    report_id: str
    message: str


class PredictRequest(BaseModel):
    model_id: str
    records: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    model_id: str
    predictions: List[Any]


class ModelListResponse(BaseModel):
    models: List[ModelSummary]


# Authentication schemas
class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class ForgotPasswordRequest(BaseModel):
    email: str


class VerifyOTPRequest(BaseModel):
    email: str
    otp_code: str


class ResetPasswordRequest(BaseModel):
    email: str
    otp_code: str
    new_password: str
    confirm_password: str


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class UserInfoResponse(BaseModel):
    user_id: str
    username: str
    email: str
    is_subscribed: bool
    free_runs_remaining: int
    total_runs: int


class SubscribeRequest(BaseModel):
    confirm: bool = True


class TrainingIntensity(BaseModel):
    level: Literal["less", "medium", "rigorous"] = "medium"


class TrainRequestWithIntensity(TrainRequest):
    intensity: Literal["less", "medium", "rigorous"] = "medium"


class TrainAllRequestWithIntensity(TrainAllRequest):
    intensity: Literal["less", "medium", "rigorous"] = "medium"


class TrainingHistoryRecord(BaseModel):
    history_id: str
    user_id: str
    dataset_id: str
    dataset_name: str
    target_column: str
    problem_type: str
    dataset_mode: Optional[str] = "csv"  # csv | supervised | unsupervised
    model_key: Optional[str] = None
    model_name: Optional[str] = None
    model_id: Optional[str] = None  # Link to model for predictions
    intensity: str
    metrics: Dict[str, Any]
    created_at: str
    report_id: Optional[str] = None
    pkl_path: Optional[str] = None  # Path to .pkl file
    h5_path: Optional[str] = None  # Path to .h5 file


class TrainingHistoryResponse(BaseModel):
    history: List[TrainingHistoryRecord]


class ModelInfoEntry(BaseModel):
    key: str
    name: str
    description: str
    problem_types: List[ProblemType]
    dataset_modes: List[str]
    advantages: List[str]
    disadvantages: List[str]
    use_cases: List[str]
    complexity: Literal["Low", "Medium", "High"]
    training_speed: Literal["Fast", "Medium", "Slow", "N/A (lazy learning)"]
    interpretability: Literal["High", "Medium", "Low"]


class ModelInfoListResponse(BaseModel):
    models: List[ModelInfoEntry]


class ModelDetailResponse(BaseModel):
    key: str
    name: str
    description: str
    problem_types: List[ProblemType]
    dataset_modes: List[str]
    advantages: List[str]
    disadvantages: List[str]
    use_cases: List[str]
    complexity: Literal["Low", "Medium", "High"]
    training_speed: Literal["Fast", "Medium", "Slow", "N/A (lazy learning)"]
    interpretability: Literal["High", "Medium", "Low"]
    how_it_works: str
    algorithm_steps: List[str]
    hyperparameters: Dict[str, str]
    visualization_type: str  # "regression_line", "decision_boundary", "tree", "feature_importance", "knn", "cnn_architecture", etc.

