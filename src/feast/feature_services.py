from feast import FeatureService

from feature_repo import *

v1_feature_service = FeatureService(
    name="model_v1",
    features=[driver_hourly_stats_view[["conv_rate"]]],
    owner="test3@gmail.com",
)

v2_feature_service = FeatureService(
    name="model_v2",
    features=[
        driver_hourly_stats_view[["conv_rate"]],
        transformed_conv_rate,
    ],
    owner="test3@gmail.com",
)

v3_feature_service = FeatureService(
    name="model_v3",
    features=[
        driver_daily_features_view,
        location_features_from_push,
    ],
    owner="test3@gmail.com",
)
