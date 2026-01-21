{{ config(materialized='table') }}

with trips as (
  select * from {{ ref('stg_trips') }}
),
route_dim as (
  select * from {{ ref('dim_route') }}
),
product_dim as (
  select * from {{ ref('dim_product') }}
)

select
  t.quote_id,

  -- foreign keys
  t.ride_hour_ts as time_key,
  r.route_key,
  p.product_key,

  -- measures
  t.price,
  t.distance,
  t.surge_multiplier,

  -- useful context
  t.timezone,
  t.latitude,
  t.longitude,

  -- keep weather in fact for now (simpler)
  t.temperature,
  t.humidity,
  t.wind_speed,
  t.visibility,
  t.short_summary

from trips t
join route_dim r
  on t.source = r.source and t.destination = r.destination
join product_dim p
  on t.cab_type = p.cab_type
 and t.product_id = p.product_id
 and t.product_name = p.product_name
