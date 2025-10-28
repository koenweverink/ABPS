# hv_from_final.py
import json
def hv_2d_max_min(points, s_ref=None, c_ref=None):
    if not points: return 0.0
    s_vals=[s for s,c in points]; c_vals=[c for s,c in points]
    s_ref = (min(s_vals)-1.0) if s_ref is None else s_ref
    c_ref = (max(c_vals)+1.0) if c_ref is None else c_ref
    pts = [(s,c) for s,c in points if (s>=s_ref and c<=c_ref)]
    pts.sort(key=lambda x: x[1])
    hv=0.0; best_s=s_ref; prev_c=c_ref
    for s,c in pts:
        if c<prev_c:
            best_s=max(best_s,s)
            hv += (prev_c-c)*(best_s-s_ref)
            prev_c=c
    return hv

data = json.load(open("robustness.json","r",encoding="utf-8"))
pts = [(d["eval"]["adjusted_raw"], d["eval"]["cost"]) for d in data["result"]["pareto_front"]]
print("Pareto size:", len(pts))
print("Hypervolume:", hv_2d_max_min(pts))
