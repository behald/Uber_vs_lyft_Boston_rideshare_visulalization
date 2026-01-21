{{ config(materialized='table') }}

select distinct
  md5(cab_type || '|' || product_id || '|' || product_name) as product_key,
  cab_type,
  product_id,
  product_name
from {{ ref('stg_trips') }}
