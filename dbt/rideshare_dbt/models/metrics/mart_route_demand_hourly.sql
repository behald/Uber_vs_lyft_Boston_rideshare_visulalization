{{ config(materialized='table') }}

select
  ride_hour_ts,
  cab_type,
  source,
  destination,
  count(*) as trips_count_last_hour,
  avg(price) as avg_price_last_hour,
  avg(price / nullif(distance, 0)) as avg_price_per_mile_last_hour,
  avg(case when surge_multiplier > 1 then 1 else 0 end) as surge_rate_last_hour
from {{ ref('stg_trips') }}
group by 1, 2, 3, 4
