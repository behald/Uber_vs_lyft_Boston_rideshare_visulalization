{{ config(materialized='table') }}

select
  cab_type,
  case when surge_multiplier > 1 then 'Surge' else 'No Surge' end as surge_flag,

  count(*) as trips_count,
  avg(price) as avg_price,
  avg(price / nullif(distance, 0)) as avg_price_per_mile

from {{ ref('stg_trips') }}
group by 1,2
