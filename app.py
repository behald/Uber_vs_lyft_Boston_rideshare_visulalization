import duckdb
import pandas as pd
import streamlit as st
import pydeck as pdk
import plotly.express as px

DB_PATH = "dbt/rideshare_dbt/dev.duckdb"

st.set_page_config(page_title="Boston Rideshare Revenue Dashboard", layout="wide")


@st.cache_resource
def get_con():
    return duckdb.connect(DB_PATH, read_only=True)


@st.cache_data
def get_table_columns(table_name: str) -> list:
    con = get_con()
    try:
        rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
        return [r[1] for r in rows]
    except Exception:
        return []


@st.cache_data
def load_mart_table(table_name: str) -> pd.DataFrame:
    con = get_con()
    return con.execute(f"SELECT * FROM {table_name}").fetchdf()


@st.cache_data
def list_marts() -> list:
    con = get_con()
    rows = con.execute(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name LIKE 'mart_%'
        ORDER BY table_name
        """
    ).fetchall()
    return [r[0] for r in rows]


def _build_where(cab_type=None, date_min=None, date_max=None, hour=None):
    where = ["1=1"]
    params = {}

    if cab_type and cab_type != "All":
        where.append("cab_type = $cab_type")
        params["cab_type"] = cab_type

    if date_min is not None:
        where.append("ride_date >= $date_min")
        params["date_min"] = str(date_min)

    if date_max is not None:
        where.append("ride_date <= $date_max")
        params["date_max"] = str(date_max)

    if hour is not None:
        where.append("ride_hour = $hour")
        params["hour"] = int(hour)

    where_sql = "WHERE " + " AND ".join(where)
    return where_sql, params


@st.cache_data
def load_map_grid(cab_type=None, date_min=None, date_max=None, hour=None):
    con = get_con()
    where_sql, params = _build_where(cab_type, date_min, date_max, hour)

    query = f"""
        SELECT
          round(latitude, 3) AS lat,
          round(longitude, 3) AS lon,
          count(*) AS trips_count,
          sum(price) AS revenue_proxy,
          avg(surge_multiplier) AS avg_surge
        FROM stg_trips
        {where_sql}
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
          AND latitude BETWEEN -90 AND 90
          AND longitude BETWEEN -180 AND 180
        GROUP BY 1,2
    """
    return con.execute(query, params).fetchdf()


@st.cache_data
def load_trips_for_analysis(cab_type=None, date_min=None, date_max=None, hour=None) -> pd.DataFrame:
    con = get_con()
    cols = set(get_table_columns("stg_trips"))

    needed = [
        "cab_type", "source", "destination",
        "ride_date", "ride_hour",
        "price", "distance", "surge_multiplier",
        "product_id", "name",
        "latitude", "longitude",
        "temperature"
    ]

    select_cols = [c for c in needed if c in cols]
    if "price" not in select_cols:
        return pd.DataFrame()

    where_sql, params = _build_where(cab_type, date_min, date_max, hour)

    query = f"""
        SELECT {", ".join(select_cols)}
        FROM stg_trips
        {where_sql}
    """
    return con.execute(query, params).fetchdf()


@st.cache_data
def load_route_arcs(cab_type=None, date_min=None, date_max=None, hour=None, limit=3, order_by="revenue"):
    con = get_con()
    where_sql, params = _build_where(cab_type, date_min, date_max, hour)
    order_sql = "revenue_proxy DESC" if order_by == "revenue" else "trips_count DESC"

    query = f"""
    WITH place_coords AS (
      SELECT
        place,
        avg(latitude) AS lat,
        avg(longitude) AS lon
      FROM (
        SELECT source AS place, latitude, longitude
        FROM stg_trips
        {where_sql}
          AND source IS NOT NULL
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
        UNION ALL
        SELECT destination AS place, latitude, longitude
        FROM stg_trips
        {where_sql}
          AND destination IS NOT NULL
          AND latitude IS NOT NULL
          AND longitude IS NOT NULL
      )
      GROUP BY 1
    ),
    top_routes AS (
      SELECT
        source,
        destination,
        count(*) AS trips_count,
        sum(price) AS revenue_proxy
      FROM stg_trips
      {where_sql}
        AND source IS NOT NULL
        AND destination IS NOT NULL
      GROUP BY 1,2
      ORDER BY {order_sql}
      LIMIT {int(limit)}
    )
    SELECT
      r.source,
      r.destination,
      r.trips_count,
      r.revenue_proxy,
      s.lat AS src_lat,
      s.lon AS src_lon,
      d.lat AS dst_lat,
      d.lon AS dst_lon
    FROM top_routes r
    JOIN place_coords s ON s.place = r.source
    JOIN place_coords d ON d.place = r.destination
    WHERE s.lat IS NOT NULL AND s.lon IS NOT NULL
      AND d.lat IS NOT NULL AND d.lon IS NOT NULL
    """
    return con.execute(query, params).fetchdf()


@st.cache_data
def load_place_labels(cab_type=None, date_min=None, date_max=None, hour=None, top_n=35, metric="revenue"):
    con = get_con()
    where_sql, params = _build_where(cab_type, date_min, date_max, hour)
    metric_sql = "sum(price) AS weight" if metric == "revenue" else "count(*) AS weight"

    query = f"""
    WITH base AS (
      SELECT source AS place, latitude, longitude, price
      FROM stg_trips
      {where_sql}
        AND source IS NOT NULL
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
      UNION ALL
      SELECT destination AS place, latitude, longitude, price
      FROM stg_trips
      {where_sql}
        AND destination IS NOT NULL
        AND latitude IS NOT NULL
        AND longitude IS NOT NULL
    ),
    agg AS (
      SELECT
        place,
        avg(latitude) AS lat,
        avg(longitude) AS lon,
        {metric_sql}
      FROM base
      GROUP BY 1
    )
    SELECT *
    FROM agg
    ORDER BY weight DESC
    LIMIT {int(top_n)}
    """
    df = con.execute(query, params).fetchdf()
    if not df.empty:
        df["label"] = df["place"]
    return df


def safe_numeric_corr(df: pd.DataFrame, target="price") -> pd.DataFrame:
    if df.empty or target not in df.columns:
        return pd.DataFrame()

    num = df.select_dtypes(include="number").copy()
    if target not in num.columns:
        return pd.DataFrame()

    keep = []
    for c in num.columns:
        if num[c].notna().mean() >= 0.6:
            keep.append(c)

    num = num[keep]
    if num.shape[1] < 2:
        return pd.DataFrame()

    return num.corr(numeric_only=True)


def smart_mart_chart(df: pd.DataFrame, title_prefix: str = ""):
    if df.empty:
        st.info("This mart table is empty.")
        return

    if title_prefix == "mart_platform_yield_comparison":
        cols = set(df.columns)
        time_col = None
        for c in ["ride_date", "date", "ds"]:
            if c in cols:
                time_col = c
                break

        metric_candidates = []
        for c in ["trips_count", "total_revenue_proxy", "avg_price", "avg_price_per_mile"]:
            if c in cols:
                metric_candidates.append(c)

        if "total_revenue" in cols and "total_revenue_proxy" not in cols:
            metric_candidates.append("total_revenue")

        if "cab_type" not in cols or not metric_candidates:
            st.info("mart_platform_yield_comparison does not have the expected columns for comparison charts.")
            return

        charts = []
        for metric in metric_candidates:
            if time_col:
                fig = px.line(
                    df,
                    x=time_col,
                    y=metric,
                    color="cab_type",
                    markers=True,
                    title=f"{metric} (Uber vs Lyft)"
                )
            else:
                agg = (
                    df.groupby("cab_type", as_index=False)[metric].sum()
                    if metric in ["trips_count", "total_revenue_proxy", "total_revenue"]
                    else df.groupby("cab_type", as_index=False)[metric].mean()
                )
                fig = px.bar(
                    agg,
                    x="cab_type",
                    y=metric,
                    title=f"{metric} (Uber vs Lyft)"
                )
            charts.append(fig)

        if len(charts) == 1:
            st.plotly_chart(charts[0], use_container_width=True)
        elif len(charts) == 2:
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts[0], use_container_width=True)
            c2.plotly_chart(charts[1], use_container_width=True)
        else:
            c1, c2 = st.columns(2)
            c1.plotly_chart(charts[0], use_container_width=True)
            c2.plotly_chart(charts[1], use_container_width=True)
            c3, c4 = st.columns(2)
            c3.plotly_chart(charts[2], use_container_width=True)
            if len(charts) > 3:
                c4.plotly_chart(charts[3], use_container_width=True)

        return

    if title_prefix == "mart_weather_revenue_impact":
        cols = set(df.columns)
        temp_bucket_col = None
        for c in [
            "temperature_bucket", "temp_bucket", "temperature_bucket_f", "temperature_bucket_F",
            "temp_bucket_f", "temp_bucket_F"
        ]:
            if c in cols:
                temp_bucket_col = c
                break

        if temp_bucket_col and "surge_rate" in cols:
            tmp = df[[temp_bucket_col, "surge_rate"] + (["cab_type"] if "cab_type" in cols else [])].copy()
            tmp[temp_bucket_col] = tmp[temp_bucket_col].astype(str)

            fig = px.bar(
                tmp,
                x=temp_bucket_col,
                y="surge_rate",
                color="cab_type" if "cab_type" in cols else None,
                barmode="group" if "cab_type" in cols else None,
                title="Surge rate by temperature bucket"
            )
            st.plotly_chart(fig, use_container_width=True)
            return

        st.info("mart_weather_revenue_impact needs temperature bucket and surge_rate columns.")
        return

    cols = df.columns.tolist()
    time_cols = [c for c in ["ride_date", "date", "ride_hour_ts", "hour_ts", "timestamp", "ds", "ride_hour"] if c in cols]
    num_cols = df.select_dtypes("number").columns.tolist()

    if "surge_flag" in cols and "cab_type" in cols and num_cols:
        metric = [c for c in num_cols if c not in ["ride_hour"]][0]
        fig = px.bar(
            df,
            x="surge_flag",
            y=metric,
            color="cab_type",
            barmode="group",
            title=f"{title_prefix} Surge vs Non-surge: {metric}"
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    if "source" in cols and "destination" in cols and num_cols:
        metric = num_cols[0]
        top_n = 15
        tmp = df.copy()
        tmp["route"] = tmp["source"].astype(str) + " → " + tmp["destination"].astype(str)
        tmp = tmp.sort_values(metric, ascending=False).head(top_n)
        fig = px.bar(
            tmp.sort_values(metric, ascending=True),
            x=metric,
            y="route",
            orientation="h",
            title=f"{title_prefix} Top routes by {metric}"
        )
        st.plotly_chart(fig, use_container_width=True)
        return

    if time_cols and num_cols:
        tcol = time_cols[0]
        metric = num_cols[0]
        if "cab_type" in cols:
            fig = px.line(
                df,
                x=tcol,
                y=metric,
                color="cab_type",
                markers=True,
                title=f"{title_prefix} {metric} over {tcol}"
            )
        else:
            fig = px.line(
                df,
                x=tcol,
                y=metric,
                markers=True,
                title=f"{title_prefix} {metric} over {tcol}"
            )
        st.plotly_chart(fig, use_container_width=True)
        return

    bucket_cols = [c for c in cols if "bucket" in c.lower() or "band" in c.lower() or "range" in c.lower()]
    if bucket_cols and num_cols:
        bcol = bucket_cols[0]
        metric = num_cols[0]
        fig = px.bar(df, x=bcol, y=metric, title=f"{title_prefix} {metric} by {bcol}")
        st.plotly_chart(fig, use_container_width=True)
        return

    if num_cols:
        metric = num_cols[0]
        fig = px.histogram(df, x=metric, title=f"{title_prefix} Distribution of {metric}")
        st.plotly_chart(fig, use_container_width=True)
        return

    st.info("No numeric columns found to chart for this mart table.")


def make_combined_routes_deck(grid, labels, arcs_demand, arcs_revenue):
    layers = []

    heatmap_layer = pdk.Layer(
        "HeatmapLayer",
        data=grid,
        get_position="[lon, lat]",
        get_weight="revenue_proxy",
        radius_pixels=60,
    )

    scatter_sample = grid.nlargest(min(len(grid), 2000), "revenue_proxy")
    scatter_layer = pdk.Layer(
        "ScatterplotLayer",
        data=scatter_sample,
        get_position="[lon, lat]",
        get_radius=50,
        pickable=True,
    )

    layers.extend([heatmap_layer, scatter_layer])

    if arcs_demand is not None and not arcs_demand.empty:
        demand_layer = pdk.Layer(
            "ArcLayer",
            data=arcs_demand,
            get_source_position="[src_lon, src_lat]",
            get_target_position="[dst_lon, dst_lat]",
            get_width=6,
            get_source_color=[255, 0, 0, 220],
            get_target_color=[255, 0, 0, 220],
            pickable=True,
        )
        layers.append(demand_layer)

    if arcs_revenue is not None and not arcs_revenue.empty:
        revenue_layer = pdk.Layer(
            "ArcLayer",
            data=arcs_revenue,
            get_source_position="[src_lon, src_lat]",
            get_target_position="[dst_lon, dst_lat]",
            get_width=6,
            get_source_color=[0, 200, 0, 220],
            get_target_color=[0, 200, 0, 220],
            pickable=True,
        )
        layers.append(revenue_layer)

    if labels is not None and not labels.empty:
        text_layer = pdk.Layer(
            "TextLayer",
            data=labels,
            get_position="[lon, lat]",
            get_text="label",
            get_size=14,
            get_color=[0, 0, 0, 200],
            get_text_anchor="'middle'",
            get_alignment_baseline="'center'",
            pickable=True,
        )
        layers.append(text_layer)

    view_state = pdk.ViewState(
        latitude=float(grid["lat"].mean()),
        longitude=float(grid["lon"].mean()),
        zoom=11,
        pitch=35
    )

    tooltip = {"text": "Route: {source} → {destination}\nTrips: {trips_count}\nRevenue: {revenue_proxy}"}

    return pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)


def _mart_explanation(mart_name: str) -> str:
    explanations = {
        "mart_platform_yield_comparison": (
            "This section compares Uber and Lyft side by side. It helps you see which platform drives more rides, "
            "more total revenue, and higher average pricing."
        ),
        "mart_weather_revenue_impact": (
            "This section connects temperature with surge behavior. It helps you see how weather changes rider demand "
            "and surge frequency."
        ),
    }
    if mart_name in explanations:
        return explanations[mart_name]

    if "route" in mart_name:
        return (
            "This section focuses on routes. It helps you see which source to destination pairs drive the most rides "
            "or the most revenue."
        )
    if "hour" in mart_name or "hourly" in mart_name:
        return (
            "This section focuses on hourly patterns. It helps you see how demand and pricing change across the day."
        )
    if "daily" in mart_name or "date" in mart_name:
        return (
            "This section focuses on daily patterns. It helps you see trend changes over time."
        )
    if "surge" in mart_name:
        return (
            "This section focuses on surge. It helps you compare surge and non surge behavior and how it affects revenue."
        )

    return (
        "This section is a dbt mart summary table. It gives a clean aggregated view so you can spot patterns faster than raw data."
    )


def _mart_insight(df: pd.DataFrame, mart_name: str) -> str:
    if df is None or df.empty:
        return "Insight: This mart returned no rows for the current filters."

    cols = set(df.columns)

    # Try to produce simple, safe insights based on common columns
    if "cab_type" in cols and "trips_count" in cols:
        g = df.groupby("cab_type", as_index=False)["trips_count"].sum().sort_values("trips_count", ascending=False)
        if len(g) >= 2:
            top = g.iloc[0]["cab_type"]
            return f"Insight: {top} has higher total ride volume in this mart view."
        return "Insight: This mart shows ride volume by platform."

    if "cab_type" in cols and ("total_revenue_proxy" in cols or "revenue_proxy" in cols or "total_revenue" in cols):
        rev_col = "total_revenue_proxy" if "total_revenue_proxy" in cols else ("total_revenue" if "total_revenue" in cols else "revenue_proxy")
        g = df.groupby("cab_type", as_index=False)[rev_col].sum().sort_values(rev_col, ascending=False)
        if len(g) >= 2:
            top = g.iloc[0]["cab_type"]
            return f"Insight: {top} generates more total revenue in this mart view."
        return "Insight: This mart shows revenue by platform."

    if "source" in cols and "destination" in cols:
        metric_cols = [c for c in ["revenue_proxy", "total_revenue_proxy", "trips_count"] if c in cols]
        if metric_cols:
            m = metric_cols[0]
            top_row = df.sort_values(m, ascending=False).iloc[0]
            return f"Insight: The strongest route here is {top_row['source']} → {top_row['destination']} based on {m}."
        return "Insight: This mart highlights route pairs. Look for the top routes in the chart."

    if any("bucket" in c.lower() for c in cols):
        return "Insight: Bucketed views help you compare demand or pricing across ranges. Look for the highest bucket bar."

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        return "Insight: The top bars or peaks in the chart show where demand or revenue concentrates in this mart view."

    return "Insight: This mart is mainly categorical. Use the preview table to understand what dimensions it summarizes."


def main():
    st.title("Boston Rideshare Revenue Dashboard (Uber vs Lyft)")
    st.caption("Built with DuckDB + dbt + Streamlit.")

    con = get_con()
    bounds = con.execute("SELECT min(ride_date), max(ride_date) FROM stg_trips").fetchone()
    min_date, max_date = bounds[0], bounds[1]

    with st.sidebar:
        st.header("Filters")
        cab_type = st.selectbox("Platform", ["All", "Uber", "Lyft"], index=0)

        date_range = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        if isinstance(date_range, tuple):
            date_min, date_max = date_range
        else:
            date_min, date_max = min_date, max_date

        hour_mode = st.checkbox("Filter by hour of day", value=False)
        hour = None
        if hour_mode:
            hour = st.slider("Hour (0–23)", 0, 23, 8)

    trips_df = load_trips_for_analysis(cab_type=cab_type, date_min=date_min, date_max=date_max, hour=hour)
    if trips_df.empty:
        st.error("No data returned from stg_trips for the current filters.")
        return

    # Add day_of_week for charts
    if "ride_date" in trips_df.columns:
        trips_df = trips_df.copy()
        dt = pd.to_datetime(trips_df["ride_date"], errors="coerce")
        trips_df["day_of_week"] = dt.dt.day_name()

    st.markdown(
        "### What this dashboard is telling us\n"
        "This dashboard explains how rides, revenue, surge, distance, time, and weather behave for Uber and Lyft. "
        "You can use the filters to see how patterns change by platform, date range, and hour."
    )

    st.divider()

    # KPIs
    st.subheader("Overall KPIs")
    st.markdown(
        "This section is a quick summary. It shows the total number of rides in the filter, the total revenue proxy, "
        "average price per mile, and the surge rate. This helps you understand the big picture before going deeper."
    )

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Rows (quotes)", f"{len(trips_df):,}")
    kpi2.metric("Revenue proxy (sum price)", f"{trips_df['price'].sum():,.0f}")

    if "distance" in trips_df.columns:
        ppm = (trips_df["price"] / trips_df["distance"].replace(0, pd.NA)).dropna()
        kpi3.metric("Avg price per mile", f"{ppm.mean():.2f}" if len(ppm) else "NA")
    else:
        kpi3.metric("Avg price per mile", "NA")

    if "surge_multiplier" in trips_df.columns:
        surge_rate = (trips_df["surge_multiplier"] > 1).mean()
        kpi4.metric("Surge rate", f"{surge_rate:.2%}")
    else:
        kpi4.metric("Surge rate", "NA")

    st.markdown(
        "Insight: Pricing is mostly driven by ride distance. Surge increases price in peak moments, but it is not the main driver across all rides."
    )

    st.divider()

    # MAPS
    st.subheader("Maps: Activity concentration and top routes")
    st.markdown(
        "This map shows where activity concentrates using a heat layer. The route lines try to show high demand routes and high revenue routes. "
        "If route lines are not visible, it is usually because the dataset does not contain true pickup and dropoff coordinates for every location name."
    )

    grid = load_map_grid(cab_type=cab_type, date_min=date_min, date_max=date_max, hour=hour)
    if grid.empty:
        st.warning("No rows with valid latitude/longitude for the current filters.")
    else:
        labels = load_place_labels(
            cab_type=cab_type,
            date_min=date_min,
            date_max=date_max,
            hour=hour,
            top_n=35,
            metric="revenue"
        )

        arcs_demand = load_route_arcs(
            cab_type=cab_type,
            date_min=date_min,
            date_max=date_max,
            hour=hour,
            limit=3,
            order_by="trips"
        )
        arcs_revenue = load_route_arcs(
            cab_type=cab_type,
            date_min=date_min,
            date_max=date_max,
            hour=hour,
            limit=3,
            order_by="revenue"
        )

        deck = make_combined_routes_deck(grid, labels, arcs_demand, arcs_revenue)
        st.pydeck_chart(deck)

    st.divider()

    # SURGE
    st.subheader("Surge analysis: Uber vs Lyft")
    st.markdown(
        "This section compares surge behavior between Uber and Lyft. It answers which platform surges more often "
        "and which platform has a higher average surge multiplier."
    )

    if "surge_multiplier" in trips_df.columns and "cab_type" in trips_df.columns:
        surge_summary = trips_df.dropna(subset=["surge_multiplier", "cab_type"]).copy()
        surge_summary["is_surge"] = surge_summary["surge_multiplier"] > 1

        agg = surge_summary.groupby("cab_type", as_index=False).agg(
            surge_rate=("is_surge", "mean"),
            avg_surge_multiplier=("surge_multiplier", "mean"),
            max_surge_multiplier=("surge_multiplier", "max"),
            rides=("surge_multiplier", "size")
        )

        c1, c2 = st.columns(2)
        with c1:
            fig_sr = px.bar(agg, x="cab_type", y="surge_rate", title="Surge rate by platform")
            st.plotly_chart(fig_sr, use_container_width=True)
        with c2:
            fig_sm = px.bar(agg, x="cab_type", y="avg_surge_multiplier", title="Avg surge multiplier by platform")
            st.plotly_chart(fig_sm, use_container_width=True)

        st.dataframe(agg, use_container_width=True, hide_index=True)

        if len(agg) >= 2:
            top_sr = agg.sort_values("surge_rate", ascending=False).iloc[0]["cab_type"]
            st.markdown(f"Insight: {top_sr} shows a higher surge rate in the selected filter range.")
        else:
            st.markdown("Insight: Surge patterns are visible in the charts above.")
    else:
        st.info("Surge comparison needs cab_type and surge_multiplier columns.")

    st.divider()

    # DAY OF WEEK
    st.subheader("Day of week impact")
    st.markdown(
        "This section shows how ride volume and revenue change across the week. It helps you see weekday commute patterns "
        "versus weekend leisure patterns."
    )

    if "day_of_week" in trips_df.columns:
        order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        if "surge_multiplier" in trips_df.columns:
            t = trips_df.dropna(subset=["day_of_week", "surge_multiplier"]).copy()

            bins = [-float("inf"), 1.0, 1.25, 1.5, float("inf")]
            labels_bins = ["1.0", "1.01-1.25", "1.26-1.50", "1.51+"]
            t["surge_bucket"] = pd.cut(t["surge_multiplier"], bins=bins, labels=labels_bins, include_lowest=True).astype(str)

            rides_dow = (
                t.groupby(["day_of_week", "surge_bucket"], as_index=False)
                 .agg(total_rides=("price", "size"))
            )
            rides_dow["day_of_week"] = pd.Categorical(rides_dow["day_of_week"], categories=order_days, ordered=True)
            rides_dow = rides_dow.sort_values("day_of_week")

            fig1 = px.bar(
                rides_dow,
                x="day_of_week",
                y="total_rides",
                color="surge_bucket",
                barmode="stack",
                title="Total rides vs day of week per surge multiplier bucket"
            )
            st.plotly_chart(fig1, use_container_width=True)

        t2 = trips_df.dropna(subset=["day_of_week", "price"]).copy()
        t2 = t2[t2["price"] > 0]

        rev_dow = (
            t2.groupby(["day_of_week"], as_index=False)
              .agg(total_price=("price", "sum"), total_rides=("price", "size"))
        )
        rev_dow["day_of_week"] = pd.Categorical(rev_dow["day_of_week"], categories=order_days, ordered=True)
        rev_dow = rev_dow.sort_values("day_of_week")

        fig2 = px.bar(rev_dow, x="day_of_week", y="total_price", title="Total price vs day of week")
        st.plotly_chart(fig2, use_container_width=True)

        peak_day = rev_dow.sort_values("total_price", ascending=False).iloc[0]["day_of_week"]
        st.markdown(f"Insight: {peak_day} shows the highest total revenue in the selected filter range.")
    else:
        st.info("Day of week charts need ride_date in stg_trips.")

    st.divider()

    # TEMPERATURE
    st.subheader("Demand vs temperature")
    st.markdown(
        "This section groups rides by temperature ranges. It helps you see whether demand goes up or down when weather changes."
    )

    if "temperature" in trips_df.columns:
        t = trips_df.dropna(subset=["temperature"]).copy()
        t["temp_bucket"] = pd.cut(t["temperature"], bins=12).astype(str)

        temp_rides = (
            t.groupby("temp_bucket", as_index=False)
             .agg(total_rides=("price", "size"), total_price=("price", "sum"))
        )

        fig_temp = px.bar(temp_rides, x="temp_bucket", y="total_rides", title="Total rides vs temperature (F) bucket")
        st.plotly_chart(fig_temp, use_container_width=True)

        if not temp_rides.empty:
            top_bucket = temp_rides.sort_values("total_rides", ascending=False).iloc[0]["temp_bucket"]
            st.markdown(f"Insight: The highest ride volume appears in the temperature bucket {top_bucket}.")
    else:
        st.info("Temperature chart needs temperature column in stg_trips.")

    st.divider()

    # AVG DISTANCE BY HOUR
    st.subheader("Average ride distance by hour")
    st.markdown(
        "This section shows the average trip distance across each hour. It helps it understand whether late night trips are longer "
        "and whether morning trips look more like short commutes."
    )

    if "ride_hour" in trips_df.columns and "distance" in trips_df.columns:
        t = trips_df.dropna(subset=["ride_hour", "distance"]).copy()
        t = t[t["distance"] > 0]

        if "cab_type" in t.columns:
            avg_dist = t.groupby(["ride_hour", "cab_type"], as_index=False).agg(avg_distance=("distance", "mean"))
            fig_dist = px.line(
                avg_dist, x="ride_hour", y="avg_distance", color="cab_type", markers=True,
                title="Average ride distance by hour (by platform)"
            )
        else:
            avg_dist = t.groupby(["ride_hour"], as_index=False).agg(avg_distance=("distance", "mean"))
            fig_dist = px.line(avg_dist, x="ride_hour", y="avg_distance", markers=True, title="Average ride distance by hour")

        st.plotly_chart(fig_dist, use_container_width=True)

        if not avg_dist.empty:
            if "cab_type" in avg_dist.columns:
                tmp = avg_dist.groupby("ride_hour", as_index=False)["avg_distance"].mean()
                top_hour = tmp.sort_values("avg_distance", ascending=False).iloc[0]["ride_hour"]
            else:
                top_hour = avg_dist.sort_values("avg_distance", ascending=False).iloc[0]["ride_hour"]
            st.markdown(f"Insight: The longest average trips appear around hour {int(top_hour)}.")
    else:
        st.info("Average distance by hour needs ride_hour and distance columns.")

    st.divider()

    # PRICE VS DISTANCE
    st.subheader("Price vs distance")
    st.markdown(
        "This section checks whether pricing increases with distance and whether price per mile stays stable. "
        "If pricing is consistent, the dots form a clean pattern instead of random spikes."
    )

    if "distance" in trips_df.columns:
        tmp = trips_df[(trips_df["distance"] > 0) & (trips_df["price"] > 0)].copy()
        tmp["price_per_mile"] = tmp["price"] / tmp["distance"]
        tmp = tmp.dropna(subset=["price_per_mile"])
        tmp_scatter = tmp.head(80000)

        c1, c2 = st.columns(2)
        with c1:
            fig_d = px.scatter(
                tmp_scatter, x="distance", y="price",
                color="cab_type" if "cab_type" in tmp_scatter.columns else None,
                title="Distance vs price (raw)"
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with c2:
            fig_pm = px.scatter(
                tmp_scatter, x="distance", y="price_per_mile",
                color="cab_type" if "cab_type" in tmp_scatter.columns else None,
                title="Distance vs price per mile (raw)"
            )
            st.plotly_chart(fig_pm, use_container_width=True)

        st.markdown(
            "Insight: Price rises with distance. Price per mile is more stable than total price, which is a good sign of consistent pricing."
        )
    else:
        st.info("Distance charts need a distance column in stg_trips.")

    st.divider()

    # CORRELATION
    st.subheader("Price correlation with continuous variables")
    st.markdown(
        "This heatmap shows which numeric variables move together with price. Strong correlation means a stronger relationship. "
        "This helps you identify the main drivers of pricing."
    )

    corr = safe_numeric_corr(trips_df, target="price")
    if corr.empty:
        st.info("Correlation needs numeric columns alongside price.")
    else:
        cols = corr.columns.tolist()
        if len(cols) > 25:
            cols = ["price"] + [c for c in corr["price"].abs().sort_values(ascending=False).head(24).index if c != "price"]
            corr = corr.loc[cols, cols]

        fig_corr = px.imshow(corr, title="Correlation heatmap (numeric columns)")
        st.plotly_chart(fig_corr, use_container_width=True)

        if "distance" in corr.columns and "price" in corr.columns:
            st.markdown("Insight: Distance is usually the strongest driver of price. Surge adds an extra layer during peak times.")
        else:
            st.markdown("Insight: The highest correlations in the heatmap point to the main drivers of price.")

    st.divider()

    # PRODUCT PERFORMANCE
    st.subheader("Product performance")
    st.markdown(
        "This section compares products. Products are ride types like shared, standard, or premium. "
        "It shows which products generate the most rides and the most revenue."
    )

    if "product_id" not in trips_df.columns:
        st.info("Product analysis needs product_id in stg_trips.")
    else:
        tmp = trips_df.copy()
        if "distance" in tmp.columns:
            tmp = tmp[tmp["distance"].fillna(0) > 0].copy()
            tmp["price_per_mile"] = tmp["price"] / tmp["distance"]
        else:
            tmp["price_per_mile"] = pd.NA

        group_cols = ["product_id"]
        if "name" in tmp.columns:
            group_cols.append("name")
        if "cab_type" in tmp.columns:
            group_cols.append("cab_type")

        prod = tmp.groupby(group_cols, as_index=False).agg(
            rides=("price", "size"),
            revenue_proxy=("price", "sum"),
            avg_price=("price", "mean"),
            avg_price_per_mile=("price_per_mile", "mean")
        ).sort_values("revenue_proxy", ascending=False)

        topn = st.slider("Top N products", 5, 40, 15)
        prod_top = prod.head(topn)

        st.dataframe(prod_top, use_container_width=True, hide_index=True)

        label_col = "name" if "name" in prod_top.columns else "product_id"
        fig_prod_rev = px.bar(
            prod_top.sort_values("revenue_proxy", ascending=True),
            x="revenue_proxy",
            y=label_col,
            color="cab_type" if "cab_type" in prod_top.columns else None,
            orientation="h",
            title="Top products by revenue proxy"
        )
        st.plotly_chart(fig_prod_rev, use_container_width=True)

        if not prod_top.empty:
            top_label = prod_top.iloc[0][label_col]
            st.markdown(f"Insight: The top product by revenue in this view is {top_label}.")
        else:
            st.markdown("Insight: Product revenue concentrates in a smaller number of products.")

    st.divider()

    # DBT MARTS (INLINE, NO DROPDOWN, NO EXPANDERS)
    st.subheader("dbt marts shown inline")
    st.markdown(
        "Below are your dbt mart tables placed directly on the page, just like the other visualizations. "
        "Each mart is an aggregated view that makes patterns easier to see than raw rows."
    )

    marts = list_marts()
    if not marts:
        st.info("No mart tables found. Run: dbt run")
        return

    remove_marts = {"mart_platform_kpis_daily", "mart_hourly_revenue"}

    for mart in marts:
        if mart in remove_marts:
            continue

        st.divider()
        st.subheader(mart)

        st.markdown(_mart_explanation(mart))

        df_mart = load_mart_table(mart)

        st.markdown("#### Chart")
        smart_mart_chart(df_mart, title_prefix=mart)

        st.markdown(_mart_insight(df_mart, mart))

        st.markdown("#### Data preview")
        st.dataframe(df_mart.head(200), use_container_width=True)

        st.download_button(
            label=f"Download {mart} as CSV",
            data=df_mart.to_csv(index=False).encode("utf-8"),
            file_name=f"{mart}.csv",
            mime="text/csv",
            key=f"dl_{mart}"
        )

    st.divider()
    st.subheader("Overall interesting findings from this dashboard")
    st.markdown(
        "First, distance is a strong driver of price, so pricing behavior looks stable and predictable. "
        "Second, surge is a secondary driver and usually shows up in peak periods, not across all rides. "
        "Third, day of week and hour patterns help explain demand cycles like commute hours and late night activity. "
        "Fourth, products and route views show that most revenue comes from a smaller set of high volume segments."
    )


if __name__ == "__main__":
    main()
