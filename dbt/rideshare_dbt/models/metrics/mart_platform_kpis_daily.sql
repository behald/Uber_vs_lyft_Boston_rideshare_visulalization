{{ config(materialized='table') }}

select
  ride_date,
  cab_type,
  count(*) as trips_count,
  avg(price) as avg_price,
  avg(distance) as avg_distance,
  avg(price / nullif(distance, 0)) as avg_price_per_mile,
  avg(case when surge_multiplier > 1 then 1 else 0 end) as surge_rate
from {{ ref('stg_trips') }}
group by 1, 2
