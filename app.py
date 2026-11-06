import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
import xarray as xr
from affine import Affine
from matplotlib import colormaps
from rasterio.features import geometry_mask

DATA_DIR = Path(__file__).parent / "data"
F_CUBO = DATA_DIR / "Cubo_NO2_Final.nc"
F_BASE = DATA_DIR / "Linea_Base_NO2_AMG.nc"
F_GEOJSON = DATA_DIR / "amg_limites.geojson"

ESC = 1e5
UNIDAD = "×10⁻⁵ mol/m²"
HOTSPOT = (14.5064, -90.6249)
CENTRO = (14.578, -90.487)
TODO_AMG = "Todo el AMG"

st.set_page_config(page_title="Dashboard de la línea base del NO₂", layout="wide")


def _extraer_dataarray(path: Path) -> xr.DataArray:
    ds = xr.open_dataset(path)
    nombre = next(v for v in ds.data_vars if ds[v].ndim >= 2)
    da = ds[nombre].load()
    ds.close()
    ren = {c: n for c, n in (("longitude", "x"), ("latitude", "y"), ("lon", "x"), ("lat", "y")) if c in da.dims}
    da = da.rename(ren)
    if "band" in da.dims:
        da = da.isel(band=0, drop=True)
    return da.drop_vars("spatial_ref", errors="ignore")


def _afin(x, y) -> Affine:
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    return Affine(dx, 0, float(x[0]) - dx / 2, 0, dy, float(y[0]) - dy / 2)


@st.cache_resource(show_spinner="Cargando cubo de datos…")
def cargar_datos():
    cubo = _extraer_dataarray(F_CUBO).transpose("time", "y", "x")
    cubo = cubo.assign_coords(time=pd.to_datetime(cubo.time.values).to_period("M").to_timestamp())
    base = _extraer_dataarray(F_BASE)
    base = base.reindex(x=cubo.x, y=cubo.y, method="nearest", tolerance=1e-4)

    cubo, base = cubo * ESC, base * ESC

    gdf = gpd.read_file(F_GEOJSON).to_crs("EPSG:4326")
    col = next((c for c in ("adm2_name", "ADM2_ES", "ADM2_EN", "NAME_2") if c in gdf.columns), None)
    if col is None:
        raise KeyError(f"No encuentro la columna de nombre municipal. Columnas: {list(gdf.columns)}")
    gdf = gdf.rename(columns={col: "municipio"})[["municipio", "geometry"]]

    x, y = cubo.x.values, cubo.y.values
    dx, dy = abs(float(x[1] - x[0])), abs(float(y[1] - y[0]))
    aff = _afin(x, y)
    mascaras = {
        m: geometry_mask([g], out_shape=(len(y), len(x)), transform=aff, all_touched=True, invert=True)
        for m, g in zip(gdf["municipio"], gdf.geometry)
    }
    mascaras[TODO_AMG] = np.any([m for m in mascaras.values()], axis=0)

    arr = cubo.values
    medias, validos = {}, {}
    for nombre, m in mascaras.items():
        sub = arr[:, m]
        validos[nombre] = np.isfinite(sub).sum(axis=1)
        if sub.shape[1] == 0:
            medias[nombre] = np.full(arr.shape[0], np.nan)
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medias[nombre] = np.nanmean(sub, axis=1)

    idx = pd.DatetimeIndex(cubo.time.values)
    return {
        "cubo": cubo,
        "base": base,
        "gdf": gdf,
        "geojson": gdf.__geo_interface__,
        "series": pd.DataFrame(medias, index=idx),
        "validos": pd.DataFrame(validos, index=idx),
        "n_pix": {k: int(m.sum()) for k, m in mascaras.items()},
        "dx": dx,
        "dy": dy,
    }


def colorear(vals, vmin, vmax, cmap):
    t = np.clip((np.asarray(vals) - vmin) / (vmax - vmin + 1e-12), 0, 1)
    rgba = (colormaps[cmap](t) * 255).astype(np.uint8)
    rgba[:, 3] = 215
    return rgba.tolist()


def barra_leyenda(vmin, vmax, cmap, titulo):
    paradas = ", ".join(
        "rgb(%d,%d,%d)" % tuple((np.array(colormaps[cmap](t))[:3] * 255).astype(int)) for t in np.linspace(0, 1, 8)
    )
    st.markdown(
        f"<div style='font-size:0.8rem'>{titulo}</div>"
        f"<div style='height:12px;border-radius:4px;background:linear-gradient(90deg,{paradas})'></div>"
        f"<div style='display:flex;justify-content:space-between;font-size:0.75rem'>"
        f"<span>{vmin:.1f}</span><span>{vmax:.1f}</span></div>",
        unsafe_allow_html=True,
    )


def pixeles_mes(d, i):
    cubo, base = d["cubo"], d["base"]
    z, b = cubo.isel(time=i).values, base.values
    X, Y = np.meshgrid(cubo.x.values, cubo.y.values)
    ok = np.isfinite(z)
    df = pd.DataFrame({"lon": X[ok], "lat": Y[ok], "val": z[ok], "anom": np.nan_to_num((z - b)[ok])})
    df["val_txt"] = df["val"].map("{:.2f}".format)
    df["anom_txt"] = df["anom"].map("{:+.2f}".format)
    return df


def construir_mapa(d, i, modo_3d, modo_valor, ver_limites, ver_hotspot, vmin, vmax, amax):
    df = pixeles_mes(d, i)
    if df.empty:
        return None
    if modo_valor == "Concentración mensual":
        df["color"] = colorear(df["val"], vmin, vmax, "YlOrRd")
    else:
        df["color"] = colorear(df["anom"], -amax, amax, "RdBu_r")

    capas = []
    if modo_3d:
        df["elev"] = df["val"] / vmax * 9000
        radio = 0.4 * min(d["dx"], d["dy"]) * 111_000
        capas.append(pdk.Layer(
            "ColumnLayer", df, get_position=["lon", "lat"], get_elevation="elev",
            elevation_scale=1, radius=radio, get_fill_color="color",
            pickable=True, auto_highlight=True, disk_resolution=16,
        ))
    else:
        hx, hy = d["dx"] / 2, d["dy"] / 2
        df["poligono"] = [
            [[lo - hx, la - hy], [lo + hx, la - hy], [lo + hx, la + hy], [lo - hx, la + hy]]
            for lo, la in zip(df["lon"], df["lat"])
        ]
        capas.append(pdk.Layer(
            "PolygonLayer", df, get_polygon="poligono", get_fill_color="color",
            stroked=False, pickable=True, auto_highlight=True,
        ))
    if ver_limites:
        capas.append(pdk.Layer(
            "GeoJsonLayer", d["geojson"], stroked=True, filled=False,
            get_line_color=[255, 255, 255, 220], line_width_min_pixels=1.2,
        ))
    if ver_hotspot:
        capas.append(pdk.Layer(
            "ScatterplotLayer", [{"lon": HOTSPOT[1], "lat": HOTSPOT[0]}],
            get_position=["lon", "lat"], get_radius=1800, stroked=True, filled=False,
            get_line_color=[0, 220, 255, 255], line_width_min_pixels=3,
        ))

    return pdk.Deck(
        layers=capas,
        initial_view_state=pdk.ViewState(latitude=CENTRO[0], longitude=CENTRO[1],
                                        zoom=9.9, pitch=55 if modo_3d else 0),
        map_style=pdk.map_styles.DARK,
        tooltip={
            "html": f"<b>NO₂:</b> {{val_txt}} {UNIDAD}<br/><b>Δ vs. línea base:</b> {{anom_txt}}",
            "style": {"backgroundColor": "#1f2937", "color": "white"},
        },
    )


def fig_serie(d, municipio, mes_sel):
    s = d["series"][municipio]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=s.index, y=s.values, mode="lines+markers", connectgaps=False, name=municipio,
        line=dict(color="#d94801", width=2), marker=dict(size=5),
        hovertemplate="%{x|%Y-%m}<br>NO₂: %{y:.2f} " + UNIDAD + "<extra></extra>",
    ))
    fig.add_hline(y=float(np.nanmean(s.values)), line_dash="dash", line_color="gray",
                annotation_text="media histórica", annotation_position="top left")
    fig.add_shape(type="line", x0=mes_sel, x1=mes_sel, y0=0, y1=1, xref="x", yref="paper",
                line=dict(color="#0ea5e9", width=2, dash="dot"))
    fig.update_layout(
        title=f"<b>Serie temporal de NO₂ · {municipio}</b>",
        yaxis_title=f"Densidad de columna [{UNIDAD}]", xaxis_title=None,
        margin=dict(l=10, r=10, t=50, b=10), height=380, hovermode="x unified",
    )
    return fig


@st.cache_data(show_spinner="Construyendo cubo espacio-temporal 3D…")
def fig_cubo_3d(vmin, vmax):
    da = cargar_datos()["cubo"]
    df = da.to_dataframe(name="no2").reset_index().dropna(subset=["no2"])
    fig = go.Figure(go.Scatter3d(
        x=df["x"], y=df["y"], z=df["time"], mode="markers",
        marker=dict(size=2.5, color=df["no2"], colorscale="YlOrRd", cmin=vmin, cmax=vmax, opacity=0.6,
                    colorbar=dict(title=f"NO₂<br>[{UNIDAD}]")),
        hovertemplate="Lon %{x:.3f}<br>Lat %{y:.3f}<br>%{z|%Y-%m}<br>NO₂: %{marker.color:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="<b>Cubo espacio-temporal 3D de NO₂ (2018-2025)</b>",
        scene=dict(xaxis_title="Longitud (°)", yaxis_title="Latitud (°)", zaxis_title="Tiempo",
                camera=dict(eye=dict(x=1.9, y=-1.9, z=1.3))),
        margin=dict(l=0, r=0, b=0, t=50), height=700,
    )
    return fig


@st.cache_data(show_spinner="Construyendo superficie…")
def fig_superficie():
    b = cargar_datos()["base"]
    fig = go.Figure(go.Surface(z=b.values, x=b.x.values, y=b.y.values, colorscale="YlOrRd",
                            colorbar=dict(title=f"[{UNIDAD}]")))
    fig.update_layout(
        title="<b>Superficie 3D de la línea base de NO₂ (2018-2025)</b>",
        scene=dict(xaxis_title="Longitud (°)", yaxis_title="Latitud (°)", zaxis_title="Densidad de columna",
                camera=dict(eye=dict(x=1.6, y=-1.6, z=1.1))),
        margin=dict(l=0, r=0, b=0, t=50), height=650,
    )
    return fig


try:
    D = cargar_datos()
except FileNotFoundError as e:
    st.error(f"No se encontró un archivo de datos: {e.filename}. Revisa la carpeta `data/`.")
    st.stop()

tiempos = list(pd.DatetimeIndex(D["cubo"].time.values))
cubo_vals = D["cubo"].values
vmin, vmax = float(np.nanpercentile(cubo_vals, 2)), float(np.nanpercentile(cubo_vals, 99))
amax = float(np.nanpercentile(np.abs(cubo_vals - D["base"].values[None]), 98))

with st.sidebar:
    st.title("Línea Base AMG")
    st.caption("Utilizando datos de Sentinel-5P · 2018–2025")
    st.markdown("**Contaminante:** Dióxido de Nitrógeno (NO₂)", unsafe_allow_html=True)
    st.divider()
    mes = st.select_slider("Mes", options=tiempos, value=tiempos[0], format_func=lambda t: t.strftime("%Y-%m"))
    modo_3d = st.radio("Vista del mapa", ["2D plano", "3D columnas"], horizontal=True) == "3D columnas"
    modo_valor = st.radio("Variable", ["Concentración mensual", "Anomalía vs. línea base"])
    ver_limites = st.checkbox("Límites municipales", value=True)
    ver_hotspot = st.checkbox("Marcar hotspot", value=False)
    st.divider()
    if modo_valor == "Concentración mensual":
        barra_leyenda(vmin, vmax, "YlOrRd", f"NO₂ [{UNIDAD}]")
    else:
        barra_leyenda(-amax, amax, "RdBu_r", f"Anomalía [{UNIDAD}]")

i = tiempos.index(mes)
tab_dash, tab_modelos = st.tabs(["Dashboard", "Cubos de datos"])

with tab_dash:
    serie_amg = D["series"][TODO_AMG]
    base_media = float(np.nanmean(D["base"].values))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"Media AMG · {mes:%Y-%m}", "sin dato" if np.isnan(serie_amg.iloc[i]) else f"{serie_amg.iloc[i]:.2f}",
            help=UNIDAD)
    mes_vals = D["cubo"].isel(time=i).values
    c2.metric("Máximo del mes", "sin dato" if np.isnan(mes_vals).all() else f"{np.nanmax(mes_vals):.2f}", help=UNIDAD)
    c3.metric("Píxeles válidos", f"{int(D['validos'][TODO_AMG].iloc[i])} / {D['n_pix'][TODO_AMG]}")
    c4.metric("Línea base AMG (media)", f"{base_media:.2f}", help=UNIDAD)

    deck = construir_mapa(D, i, modo_3d, modo_valor, ver_limites, ver_hotspot, vmin, vmax, amax)
    if deck is None:
        st.warning("Este mes no tiene píxeles válidos.")
    else:
        st.pydeck_chart(deck)

    st.subheader("Serie temporal por municipio")
    nombres = [TODO_AMG] + sorted(D["gdf"]["municipio"].unique())
    muni = st.selectbox("Municipio", nombres)
    st.plotly_chart(fig_serie(D, muni, mes))

    s = D["series"][muni]
    v = D["validos"][muni]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Media histórica", f"{np.nanmean(s.values):.2f}" if s.notna().any() else "—", help=UNIDAD)
    k2.metric("Máximo mensual", f"{np.nanmax(s.values):.2f}" if s.notna().any() else "—",
            help=f"{s.idxmax():%Y-%m}" if s.notna().any() else None)
    k3.metric("Meses sin datos", f"{int(s.isna().sum())} de {len(s)}")
    k4.metric("Píxeles que tocan el municipio", D["n_pix"][muni])
    if D["n_pix"][muni] < 4 and muni != TODO_AMG:
        st.caption("Municipio pequeño frente a la resolución del sensor.")

with tab_modelos:
    vista = st.radio("Modelo", ["Cubo espacio-temporal 3D", "Promedio temporal"], horizontal=True)
    if vista == "Cubo espacio-temporal 3D":
        st.caption("Cada punto es un píxel válido en un mes; el eje vertical es el tiempo. Arrastra para rotar.")
        st.plotly_chart(fig_cubo_3d(vmin, vmax))
    else:
        b = D["base"]
        iy, ix = np.unravel_index(np.nanargmax(b.values), b.shape)
        st.caption(f"Máximo de la línea base: {np.nanmax(b.values):.2f} {UNIDAD} en "
                f"Lat {float(b.y[iy]):.4f}, Lon {float(b.x[ix]):.4f}.")
        st.plotly_chart(fig_superficie())
