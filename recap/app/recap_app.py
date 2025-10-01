# recap_app.py
import os
import streamlit as st
import geopandas as gpd
import folium
from streamlit_folium import st_folium
from shapely import wkt
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.calibration import calibration_curve
import altair as alt

# --------------------------
# Config / constants
# --------------------------
INDEX_FILE = "info/predictions.csv"   # file your app uses
GRADCAM_DIR = "info/gradcam_examples"
LABEL_COLORS = {
    "no-damage": "green",
    "minor-damage": "yellow",
    "major-damage": "orange",
    "destroyed": "red",
}
SEVERITY_ORDER = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}

os.makedirs(GRADCAM_DIR, exist_ok=True)

# --------------------------
# Helpers
# --------------------------
def sanitize_for_display(df, max_rows=None, drop_geometry=True):
    """
    Return a pandas DataFrame safe for Streamlit/Arrow:
     - converts geometry column to string OR drops it
     - optionally truncates to max_rows to avoid huge serializations
    """
    df2 = df.copy()
    if "geometry" in df2.columns:
        if drop_geometry:
            df2 = df2.drop(columns=["geometry"])
        else:
            df2["geometry"] = df2["geometry"].astype(str)
    if max_rows is not None:
        return df2.head(max_rows)
    return df2

# --------------------------
# Load predictions (cached)
# --------------------------
@st.cache_data
def load_predictions(path):
    df = pd.read_csv(path)
    # Ensure polygon geometry column exists (may be string)
    if "polygon_wkt" in df.columns:
        try:
            df["geometry"] = df["polygon_wkt"].apply(lambda x: wkt.loads(x) if pd.notna(x) else None)
        except Exception:
            # fallback: keep geometry None if parsing fails
            df["geometry"] = None
    else:
        df["geometry"] = None

    # return GeoDataFrame for map computations
    try:
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
    except Exception:
        df["geometry"] = None
        gdf = gpd.GeoDataFrame(df, geometry="geometry")
    return gdf

gdf = load_predictions(INDEX_FILE)

# Build mapping label_id -> label_name (ground truth)
label_id_map = {}
if "label_id" in gdf.columns and "label_name" in gdf.columns:
    try:
        tmp = gdf[["label_id","label_name"]].drop_duplicates()
        label_id_map = dict(zip(tmp["label_id"], tmp["label_name"]))
    except Exception:
        label_id_map = {}

# --------------------------
# App UI
# --------------------------
st.set_page_config(layout="wide")
st.title("🛰️ RECAP – Rapid Event-level Classification of Affected Properties")
st.markdown("Prototype app: building-level damage predictions — Week 8 polish edition")

st.sidebar.header("App Sections")
section = st.sidebar.radio("Go to", ["Map", "Inspection Queue", "Inspector", "Metrics"])

# Event selector in sidebar
if "event" in gdf.columns:
    events = list(gdf["event"].dropna().unique())
    events.sort()
    event = st.sidebar.selectbox("Select Event", ["All"] + events)
    if event != "All":
        working = gdf[gdf["event"] == event].copy()
    else:
        working = gdf.copy()
else:
    working = gdf.copy()

# Confidence slider
conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
if "label_conf" in working.columns:
    working = working[working["label_conf"] >= conf_thresh]

# Helper: map predicted numeric class to name via label_id_map (best-effort)
def map_pred_to_name(row):
    try:
        if pd.isna(row.get("label_pred")):
            return None
        pred = int(row["label_pred"])
        return label_id_map.get(pred, str(pred))
    except Exception:
        return None

working["pred_label_name"] = working.apply(lambda r: map_pred_to_name(r), axis=1)

# Priority computation for inspection queue
def compute_priority(df, severity_order=SEVERITY_ORDER, alpha=0.7):
    df = df.copy()
    def sev(row):
        lname = row.get("pred_label_name") or row.get("label_name")
        return severity_order.get(lname, 0)
    df["severity_score"] = df.apply(sev, axis=1).astype(float)
    df["uncertainty"] = 1.0 - df.get("label_conf", 1.0)
    df["priority"] = df["severity_score"] * alpha + df["uncertainty"] * (1 - alpha)
    df = df.sort_values(["priority", "severity_score", "label_conf"], ascending=[False, False, True])
    return df

# --------------------------
# LEGEND HTML (used by folium)
# --------------------------
legend_html = """
<div style="
    position: fixed; 
    bottom: 50px; left: 50px; width: 180px; 
    background-color: white; 
    border:2px solid grey; 
    z-index:9999; 
    font-size:14px;
    color: black;
    padding: 10px;
">
<b>Damage Legend</b><br>
<div style="display:flex; align-items:center;"><div style="width:15px;height:15px;background:green;margin-right:5px;"></div>No damage</div>
<div style="display:flex; align-items:center;"><div style="width:15px;height:15px;background:yellow;margin-right:5px;"></div>Minor</div>
<div style="display:flex; align-items:center;"><div style="width:15px;height:15px;background:orange;margin-right:5px;"></div>Major</div>
<div style="display:flex; align-items:center;"><div style="width:15px;height:15px;background:red;margin-right:5px;"></div>Destroyed</div>
</div>
"""

# --------------------------
# MAP Section (auto-detect pixel vs lat/lon)
# --------------------------
if section == "Map":
    st.header("Map view / Pixel-space scatter (auto-detect coordinates)")
    if working.empty:
        st.warning("⚠️ No buildings match the current filter (try lowering confidence).")
    else:
        # compute centroids if geometry present
        def centroid_xy(geom):
            try:
                if geom is None:
                    return None, None
                c = geom.centroid
                return float(c.x), float(c.y)
            except Exception:
                return None, None

        centroids = working["geometry"].apply(lambda g: centroid_xy(g) if g is not None else (None, None))
        xs = [c[0] for c in centroids]
        ys = [c[1] for c in centroids]

        def looks_like_latlon(xs, ys):
            xsf = [x for x in xs if x is not None and not np.isnan(x)]
            ysf = [y for y in ys if y is not None and not np.isnan(y)]
            if len(xsf) < 10:
                return False
            lon_ok = all((-180.0 <= x <= 180.0) for x in xsf[:min(len(xsf), 200)])
            lat_ok = all((-90.0 <= y <= 90.0) for y in ysf[:min(len(ysf), 200)])
            var_ok = (np.std(xsf[:min(len(xsf),200)]) > 1e-6) and (np.std(ysf[:min(len(ysf),200)]) > 1e-6)
            return lon_ok and lat_ok and var_ok

        is_geo = looks_like_latlon(xs, ys)

        if is_geo:
            try:
                # Use folium for geographic data
                cx = working.geometry.centroid.y.mean()
                cy = working.geometry.centroid.x.mean()
                m = folium.Map(location=[cx, cy], zoom_start=4, tiles="cartodbpositron")
                for _, row in working.iterrows():
                    geom = row.get("geometry")
                    if geom is None or geom.is_empty:
                        continue
                    centroid = geom.centroid
                    label = row.get("label_name", "unknown")
                    color = LABEL_COLORS.get(label, "gray")
                    conf = row.get("label_conf", 1.0)
                    folium.CircleMarker(
                        location=[centroid.y, centroid.x],
                        radius=3,
                        color=color,
                        fill=True,
                        fill_opacity=0.7,
                        tooltip=f"{label} ({conf:.2f})"
                    ).add_to(m)
                m.get_root().html.add_child(folium.Element(legend_html))
                st_folium(m, width=800, height=500)
            except Exception as e:
                st.error(f"Failed to render folium map: {e}")
        else:
            st.info("Detected pixel/image coordinates — plotting in pixel space. Switches to world map when lat/lon are available.")
            df_plot = working.copy()
            df_plot["centroid_x"] = df_plot["geometry"].apply(lambda g: float(g.centroid.x) if g is not None else np.nan)
            df_plot["centroid_y"] = df_plot["geometry"].apply(lambda g: float(g.centroid.y) if g is not None else np.nan)
            df_plot = df_plot.dropna(subset=["centroid_x","centroid_y"])
            if df_plot.shape[0] == 0:
                st.warning("No valid centroid coordinates to plot.")
            else:
                df_plot["centroid_x"] = df_plot["centroid_x"].astype(float)
                df_plot["centroid_y"] = df_plot["centroid_y"].astype(float)
                df_plot["uncertainty"] = 1.0 - df_plot.get("label_conf", 1.0)
                df_plot["size"] = (df_plot["uncertainty"].fillna(0.0) * 30) + 10

                # drop geometry before passing to Altair
                df_plot_display = sanitize_for_display(df_plot, drop_geometry=True)

                chart = alt.Chart(df_plot_display).mark_circle(opacity=0.8).encode(
                    x=alt.X("centroid_x:Q", title="pixel x"),
                    y=alt.Y("centroid_y:Q", title="pixel y", scale=alt.Scale(reverse=True)),
                    color=alt.Color("label_name:N", scale=alt.Scale(domain=list(LABEL_COLORS.keys()),
                                                                    range=list(LABEL_COLORS.values())),
                                    legend=alt.Legend(title="Label")),
                    size=alt.Size("size:Q", legend=None),
                    tooltip=["building_id", "event", "label_name", alt.Tooltip("label_conf:Q", format=".3f")]
                ).interactive()

                st.altair_chart(chart.properties(width=900, height=500), use_container_width=False)

                if st.checkbox("Show top 20 by uncertainty"):
                    top_uncertain = df_plot.sort_values("uncertainty", ascending=False).head(20)[["building_id","event","label_name","label_conf","uncertainty"]]
                    st.dataframe(sanitize_for_display(top_uncertain, max_rows=20, drop_geometry=True))

# --------------------------
# INSPECTION QUEUE
# --------------------------
elif section == "Inspection Queue":
    st.header("Inspection Queue — ranked triage")
    if working.empty:
        st.warning("No buildings available for queue (check filters).")
    else:
        queue = compute_priority(working, severity_order=SEVERITY_ORDER, alpha=0.7)
        n = st.number_input("Show top N", min_value=5, max_value=1000, value=50, step=5)
        display_cols = ["building_id", "event", "label_name", "pred_label_name", "label_conf", "severity_score", "priority"]
        available = [c for c in display_cols if c in queue.columns]
        top = queue[available].head(n).reset_index(drop=True)
        top_display = sanitize_for_display(top, max_rows=500, drop_geometry=True)
        st.dataframe(top_display)

        st.markdown("**Queue controls**")
        if st.button("Download queue CSV"):
            csv_bytes = top_display.to_csv(index=False).encode("utf-8")
            st.download_button("Download CSV", csv_bytes, file_name="inspection_queue.csv")

# --------------------------
# INSPECTOR (with Grad-CAM)
# --------------------------
elif section == "Inspector":
    st.header("Inspector — pre/post images and Grad-CAM")
    if working.empty:
        st.warning("⚠️ No buildings to inspect at this confidence threshold.")
    else:
        queue = compute_priority(working, severity_order=SEVERITY_ORDER)
        col1, col2 = st.columns([1,2])
        with col1:
            if st.button("Pick top priority building"):
                selected_id = str(queue.iloc[0]["building_id"])
            else:
                selected_id = st.selectbox("Select Building ID", working["building_id"].astype(str).unique())
        with col2:
            st.write("Tip: use the queue to pick urgent items.")

        row = working[working["building_id"].astype(str) == str(selected_id)]
        if row.empty:
            st.error("Building not found in current filtered set.")
        else:
            row = row.iloc[0]
            st.subheader(f"Building: {row['building_id']} (Event: {row.get('event', '')})")
            st.write(f"**True label:** {row.get('label_name','-')}")
            pred_name = row.get("pred_label_name") or f"{row.get('label_pred')}"
            st.write(f"**Predicted:** {pred_name} — confidence: {row.get('label_conf', 0.0):.2f}")

            # load pre/post safely
            pre_path = row.get("pre_path")
            post_path = row.get("post_path")
            pre_img = None
            post_img = None
            if isinstance(pre_path, str) and os.path.exists(pre_path):
                pre_img = cv2.imread(pre_path, cv2.IMREAD_COLOR)
                pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
            if isinstance(post_path, str) and os.path.exists(post_path):
                post_img = cv2.imread(post_path, cv2.IMREAD_COLOR)
                post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)

            if pre_img is None or post_img is None:
                st.error("Could not load pre/post images for this building (paths might be local to your machine).")
            else:
                fig, axes = plt.subplots(1, 2, figsize=(8,4))
                axes[0].imshow(pre_img); axes[0].set_title("Pre-disaster"); axes[0].axis("off")
                axes[1].imshow(post_img); axes[1].set_title("Post-disaster"); axes[1].axis("off")
                st.pyplot(fig)

            # Grad-CAM overlay if available
            gradcam_path = os.path.join(GRADCAM_DIR, f"{row['building_id']}_overlay.png")
            alt_names = [
                os.path.join(GRADCAM_DIR, f"{row['building_id']}_pred{int(row.get('label_pred',0))}_true{int(row.get('label_id',0))}.png"),
                os.path.join(GRADCAM_DIR, f"building_{row['building_id']}_pred{int(row.get('label_pred',0))}_true{int(row.get('label_id',0))}.png"),
            ]
            overlay_found = None
            if os.path.exists(gradcam_path):
                overlay_found = gradcam_path
            else:
                for p in alt_names:
                    if os.path.exists(p):
                        overlay_found = p
                        break

            if overlay_found:
                st.subheader("Grad-CAM overlay")
                st.image(overlay_found, use_column_width=True)
            else:
                st.info("No Grad-CAM overlay found for this building. To enable overlays, generate and save them to info/gradcam_examples/.")

            # show a small metadata table (no geometry)
            meta_df = pd.DataFrame([row.to_dict()])
            st.table(sanitize_for_display(meta_df, drop_geometry=True))

# --------------------------
# METRICS Tab
# --------------------------
elif section == "Metrics":
    st.header("Evaluation metrics")
    if working.empty:
        st.warning("No data to show metrics for (adjust filters).")
    else:
        if "label_name" not in working.columns or "label_pred" not in working.columns:
            st.warning("Missing label_name or label_pred columns required for metrics.")
        else:
            preds_unique = sorted(working["label_pred"].dropna().unique().tolist())
            pred_name_map = {p: label_id_map.get(int(p), str(int(p))) for p in preds_unique}

            y_true_names = working["label_name"].astype(str).tolist()
            y_pred_names = []
            for _, r in working.iterrows():
                try:
                    pn = pred_name_map.get(int(r["label_pred"]), None)
                except Exception:
                    pn = None
                if pn is None:
                    pn = r.get("pred_label_name") or str(int(r["label_pred"]))
                y_pred_names.append(pn)

            labels = sorted(list(set(y_true_names) | set(y_pred_names)), key=lambda x: SEVERITY_ORDER.get(x, 0))
            cm = confusion_matrix(y_true_names, y_pred_names, labels=labels)
            fig, ax = plt.subplots(figsize=(6,5))
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
            for i in range(len(labels)):
                for j in range(len(labels)):
                    ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i,j] > cm.max()/2 else "black")
            ax.set_title("Confusion matrix (rows=true, cols=pred)")
            st.pyplot(fig)

            p, r, f1, sup = precision_recall_fscore_support(y_true_names, y_pred_names, labels=labels, zero_division=0)
            df_metrics = pd.DataFrame({"label": labels, "precision": p, "recall": r, "f1": f1, "support": sup})
            st.subheader("Per-class metrics")
            st.dataframe(sanitize_for_display(df_metrics, drop_geometry=True))

            macro_f1 = np.nanmean(f1)
            st.write(f"**Macro F1:** {macro_f1:.4f}")

            # Calibration
            st.subheader("Calibration (predicted-class confidences)")
            if "label_conf" not in working.columns:
                st.info("No label_conf column available for calibration.")
            else:
                fig2, ax2 = plt.subplots(figsize=(6,5))
                for pid in preds_unique:
                    try:
                        pid_int = int(pid)
                    except Exception:
                        continue
                    mask = working["label_pred"].astype(int) == pid_int
                    if mask.sum() < 10:
                        continue
                    y_true_bin = (working.loc[mask, "label_id"].astype(int) == pid_int).astype(int).values
                    y_prob = working.loc[mask, "label_conf"].values
                    frac_pos, mean_pred = calibration_curve(y_true_bin, y_prob, n_bins=10)
                    ax2.plot(mean_pred, frac_pos, marker="o", label=pred_name_map.get(pid_int, str(pid_int)))
                ax2.plot([0,1],[0,1],"k--")
                ax2.set_xlabel("Mean predicted probability")
                ax2.set_ylabel("Fraction positive")
                ax2.set_title("Calibration curve (one-curve per predicted class)")
                ax2.legend()
                st.pyplot(fig2)

                def compute_ece(y_true_bin, y_prob, n_bins=10):
                    bins = np.linspace(0,1,n_bins+1)
                    ece = 0.0
                    for i in range(n_bins):
                        mask = (y_prob > bins[i]) & (y_prob <= bins[i+1])
                        if mask.sum() == 0: continue
                        acc = y_true_bin[mask].mean()
                        conf = y_prob[mask].mean()
                        ece += (mask.sum()/len(y_prob)) * abs(acc - conf)
                    return ece
                ece_results = {}
                for pid in preds_unique:
                    try:
                        pid_int = int(pid)
                    except Exception:
                        continue
                    mask = working["label_pred"].astype(int) == pid_int
                    if mask.sum() < 10:
                        ece_results[pred_name_map.get(pid_int, str(pid_int))] = None
                    else:
                        y_true_bin = (working.loc[mask, "label_id"].astype(int) == pid_int).astype(int).values
                        y_prob = working.loc[mask, "label_conf"].values
                        ece_results[pred_name_map.get(pid_int, str(pid_int))] = compute_ece(y_true_bin, y_prob)
                st.write("Estimated ECE (predicted-class confidences):")
                st.json(ece_results)

            if st.button("Download filtered predictions"):
                csv = sanitize_for_display(working, drop_geometry=True).to_csv(index=False)
                st.download_button("Download CSV", csv.encode("utf-8"), file_name=f"predictions_filtered_{event if 'event' in locals() else 'all'}.csv")

# --------------------------
# End
# --------------------------
