import os
import shutil
from agri_ai import predict_leaf, predict_tree

print("==================================================================")
print("       AGRIGUARD AI: INDEPENDENT SCANNER VERIFICATION SUITE       ")
print("==================================================================")

# 1. Test Leaf Scanner with generic camera filenames (ZERO filename hints)
test_cases_leaf = [
    ("static/samples/sample_healthy_leaf.jpg", "camera_leaf_scan_1.jpg", "Healthy Foliage (No Disease Detected)", True),
    ("static/samples/sample_corn_rust.jpg", "camera_leaf_scan_2.jpg", "Common Rust", False),
    ("static/samples/sample_potato_late_blight.jpg", "camera_leaf_scan_3.jpg", "Late Blight", False),
    ("static/samples/sample_tomato_early_blight.jpg", "camera_leaf_scan_4.jpg", "Early Blight", False),
    ("static/samples/sample_apple_scab.jpg", "camera_leaf_scan_5.jpg", "Apple Scab", False),
    ("static/samples/sample_rice_blast.jpg", "camera_leaf_scan_6.jpg", "Rice Blast", False),
]

print("\n--- 1. CROP LEAF SCANNER: CAMERA SNAPSHOTS (NO FILENAME HINTS) ---")
leaf_passes = 0
for src, cam_name, expected_disease, expected_healthy in test_cases_leaf:
    dest = os.path.join("static", "uploads", cam_name)
    shutil.copyfile(src, dest)
    try:
        res = predict_leaf(dest, "en")
        d_name = res["disease_name"]
        is_healthy = res["is_healthy"]
        conf = res["confidence"]
        
        ok = (expected_disease in d_name) and (is_healthy == expected_healthy)
        status_tag = "[PASS]" if ok else "[FAIL]"
        if ok:
            leaf_passes += 1
        print(f" {status_tag} {cam_name:22} -> Identified: {d_name:<36} (Conf: {conf}%, Healthy: {is_healthy})")
    finally:
        if os.path.exists(dest):
            os.remove(dest)

print(f"-> Crop Scanner Score: {leaf_passes}/{len(test_cases_leaf)} Passed.")

# 2. Test Tree Scanner with generic camera filenames (ZERO filename hints)
test_cases_tree = [
    ("static/samples/sample_tree_teak.jpg", "camera_tree_scan_1.jpg", "teak", "Teak"),
    ("static/samples/sample_tree_mango.jpg", "camera_tree_scan_2.jpg", "mango", "Mango"),
    ("static/samples/sample_tree_banyan.jpg", "camera_tree_scan_3.jpg", "banyan", "Banyan"),
    ("static/samples/sample_tree_neem.jpg", "camera_tree_scan_4.jpg", "neem", "Neem"),
    ("static/samples/sample_tree_sandalwood.jpg", "camera_tree_scan_5.jpg", "sandalwood", "Sandalwood"),
]

print("\n--- 2. TREE SCANNER: CAMERA SNAPSHOTS (NO FILENAME HINTS) ---")
tree_passes = 0
for src, cam_name, expected_sp, expected_common in test_cases_tree:
    dest = os.path.join("static", "uploads", cam_name)
    shutil.copyfile(src, dest)
    try:
        res = predict_tree(dest, "en")
        sp = res["species"]
        common = res["common_name"]
        cat = res["category"]
        conf = res["confidence"]
        
        ok = (sp == expected_sp)
        status_tag = "[PASS]" if ok else "[FAIL]"
        if ok:
            tree_passes += 1
        print(f" {status_tag} {cam_name:22} -> Identified: {sp:<12} ({common}) | Cat: {cat:<10} | Conf: {conf}%")
    finally:
        if os.path.exists(dest):
            os.remove(dest)

print(f"-> Tree Scanner Score: {tree_passes}/{len(test_cases_tree)} Passed.")

# 3. Test Crop Selector Integration
print("\n--- 3. CROP SCANNER: 200-CROPS TARGET SELECTION ---")
crop_tests = [
    ("rice", "static/samples/sample_healthy_leaf.jpg", True),
    ("rice", "static/samples/sample_rice_blast.jpg", False),
    ("tomato", "static/samples/sample_tomato_early_blight.jpg", False),
    ("potato", "static/samples/sample_healthy_leaf.jpg", True),
]

selector_passes = 0
for cid, src, exp_healthy in crop_tests:
    res = predict_leaf(src, "en", crop_id=cid)
    ok = (res["is_healthy"] == exp_healthy) and (cid in res["crop"].lower())
    status_tag = "[PASS]" if ok else "[FAIL]"
    if ok:
        selector_passes += 1
    print(f" {status_tag} Target Crop: {cid:<8} -> Crop: {res['crop']:<12} | Diagnosis: {res['disease_name']:<36} | Healthy: {res['is_healthy']}")

print(f"-> Crop Selector Score: {selector_passes}/{len(crop_tests)} Passed.")

print("\n==================================================================")
all_ok = (leaf_passes == len(test_cases_leaf)) and (tree_passes == len(test_cases_tree)) and (selector_passes == len(crop_tests))
if all_ok:
    print(" ALL SCANNERS VALIDATED: 100% WORKING & IDENTIFYING ACCURATELY! ")
else:
    print(" WARNING: Some cases did not match expected output.")
print("==================================================================")
