{{ config(materialized='table') }}

select distinct
  ride_hour_ts as time_key,
  ride_hour_ts,
  ride_date,
  ride_hour,
  extract('year' from ride_date) as ride_year,
  extract('month' from ride_date) as ride_month,
  extract('day' from ride_date) as ride_day,
  extract('dow' from ride_date) as day_of_week,
  case when extract('dow' from ride_date) in (0, 6) then true else false end as is_weekend
from {{ ref('stg_trips') }}
