{{ config(materialized='view') }}

select *
from {{ source('ops', 'mart_driver_reco_hourly') }}
