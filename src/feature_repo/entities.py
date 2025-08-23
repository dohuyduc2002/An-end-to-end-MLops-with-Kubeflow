from feast import Entity, ValueType, Project

project = Project(name="homecredit")

# Spark entities
application_entity = Entity(
    name="application", join_keys=["sk_id_curr"], value_type=ValueType.STRING
)

# Kafka entities
bureau_balance_entity = Entity(
    name="bureau-balance", join_keys=["sk_id_curr"], value_type=ValueType.STRING
)
