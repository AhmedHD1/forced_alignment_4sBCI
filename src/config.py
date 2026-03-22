# Panel dimensions (mm) for all figure panels.
# Imported by src/utils.py and available to all notebooks via `from src.utils import *`.
# Edit this file to change panel sizes across the entire project.

PLOT_STYLE = {
    "box": {
        "marker_size":    2,
        "marker_opacity": 0.55,
        "line_width":     0.5,
        "jitter":         0.4,
        "pointpos":       -2.0,
        "boxmean":        True,
        "boxgap":         0.65,
        "boxgap_hue":     0.45,   # boxgap for hue-only plots (no x=)
        "boxgroupgap":    0.25,   # gap between x-groups (patients)
        "box_spacing":    0.30,   # center-to-center spacing between hued boxes within a group (go.Box path)
        "box_width":      0.22,   # explicit box width for go.Box path (controls dot-box separation)
        "group_spacing":  0.75,   # x-axis distance between patient group centers (go.Box path)
    },
    "font": {
        "family":      "Arial",
        "size_tick":   5,
        "size_label":  6,
        "size_title":  7,
    },
}

PANEL_DIMS = {
    "figure1": {
        "audio": {"width_mm": 80, "height_mm": 30},
        "trace": {"width_mm": 80, "height_mm": 30},
    },
    "figure2": {
        "all_patients": {"width_mm": 75, "height_mm": 70},
        "by_patient":   {"width_mm": 95, "height_mm": 70},
        "by_position":  {"width_mm": 75, "height_mm": 70},
        "by_phoneme":   {"width_mm": 95, "height_mm": 70},
    },
    "figure3": {
        "traces":                 {"width_mm": 80, "height_mm": 30},
        "tsne_scatter":           {"width_mm": 80, "height_mm": 80},
        "tsne_dist_s1":           {"width_mm": 60, "height_mm": 70},
        "tsne_dist_all_patients": {"width_mm": 80, "height_mm": 70},
    },
    "figure4": {
        "line_by_patient": {"width_mm": 60, "height_mm": 70},
        "by_position":     {"width_mm": 60, "height_mm": 70},
        "by_patient":      {"width_mm": 80, "height_mm": 70},
        "dtw_line":        {"width_mm": 80, "height_mm": 70},
    },
}
