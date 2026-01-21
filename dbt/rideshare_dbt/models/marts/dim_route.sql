{{ config(materialized='table') }}

select distinct
  md5(source || '|' || destination) as route_key,
  source,
  destination
from {{ ref('stg_trips') }}
