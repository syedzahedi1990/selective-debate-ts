"""Non-LLM arbitration baselines and routers.

A *baseline* picks a forecast (or ensemble policy) without any router. A
*router* predicts failure risk on each instance and routes between cheap/
expensive decision modes.
"""
from sdfts.routers.baselines import (
    validation_best,
    horizonwise_validation_best,
    simple_mean_ensemble,
    median_ensemble,
    validation_weighted_ensemble,
)
from sdfts.routers.featurize import card_to_router_features
from sdfts.routers.train_router import train_router, save_router, load_router
from sdfts.routers.predict_router import predict_failure_prob

__all__ = [
    "validation_best",
    "horizonwise_validation_best",
    "simple_mean_ensemble",
    "median_ensemble",
    "validation_weighted_ensemble",
    "card_to_router_features",
    "train_router",
    "save_router",
    "load_router",
    "predict_failure_prob",
]
