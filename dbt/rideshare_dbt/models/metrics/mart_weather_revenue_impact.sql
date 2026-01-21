{{ config(materialized='table') }}

select
  cab_type,

  case
    when temperature < 32 then 'Very Cold'
    when temperature between 32 and 50 then 'Cold'
    when temperature between 51 and 70 then 'Mild'
    else 'Warm'
  end as temperature_bucket,

  avg(price) as avg_price,
  avg(price / nullif(distance, 0)) as avg_price_per_mile,
  avg(case when surge_multiplier > 1 then 1 else 0 end) as surge_rate

from {{ ref('stg_trips') }}
where temperature is not null
group by 1,2
