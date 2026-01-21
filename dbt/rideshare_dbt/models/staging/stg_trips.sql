{{ config(materialized='table') }}

select
  quote_id,
  ride_ts,
  ride_hour_ts,
  ride_date,
  ride_hour,

  cab_type,
  product_id,
  product_name,

  source,
  destination,

  price,
  distance,
  surge_multiplier,

  timezone,
  latitude,
  longitude,

  temperature,
  humidity,
  wind_speed,
  visibility,
  short_summary

from read_parquet('../../data/silver/trips/**/*.parquet')
where price >= 0
  and distance > 0
  and cab_type in ('Uber', 'Lyft')
  and source is not null
  and destination is not null
