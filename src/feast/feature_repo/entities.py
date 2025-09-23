from feast import Entity, ValueType, Project

project = Project(name="homecredit")

# Spark entities
gold_fact_loan_entity = Entity(
    name="application", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_fact_bureau_balance_entity = Entity(
    name="bureau_balance", join_keys=["sk_id_curr","sk_id_bureau"], value_type=ValueType.INT64
)

gold_dim_demographic_entity = Entity(
    name="demographic", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_user_contract_entity = Entity(
    name="user_contract", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_user_region_entity = Entity(
    name="user_region", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_asset_and_bureau_entity = Entity(
    name="asset_and_bureau", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_user_income_entity = Entity(
    name="user_income", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_external_source_entity = Entity(
    name="external_source", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_application_time_entity = Entity(
    name="application_time", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_provided_docs_entity = Entity(
    name="provided_docs", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)

gold_dim_aggregated_entity = Entity(
    name="aggregated", join_keys=["sk_id_curr"], value_type=ValueType.INT64
)


# Kafka entities
stream_bureau_balance_entity = Entity(
    name="bureau_balance", join_keys=["sk_id_curr","sk_id_bureau"], value_type=ValueType.INT64
)
