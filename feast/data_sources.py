from feast import BigQuerySource

driver_stats = BigQuerySource(
    name="driver_stats_source",
    table="mlops-465414.feast_bq_dataset.driver_stats",  # format: project.dataset.table
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
    description="Stats of a driver based on hourly logs",
    owner="test2@gmail.com",
)
