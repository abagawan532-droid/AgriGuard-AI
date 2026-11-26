import unittest
from app import app

class AgriGuardAiTestSuite(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True

    def test_homepage(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AgriGuard", response.data)
        self.assertIn(b"200 CROPS", response.data)
        print("[OK] Homepage test passed")

    def test_crops_catalog(self):
        response = self.client.get("/crops")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"200 Comprehensive Crops Catalog", response.data)
        self.assertIn(b"Rice", response.data)
        self.assertIn(b"Ragi", response.data)
        print("[OK] 200 Crops Catalog test passed")

    def test_crops_filtering(self):
        response = self.client.get("/crops?category=cereals_millets")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Showing <b>25</b>", response.data)
        print("[OK] Crops Category Filtering test passed (25 cereals)")

    def test_crop_detail(self):
        response = self.client.get("/crop/rice")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Oryza sativa", response.data)
        self.assertIn(b"Blast", response.data)
        print("[OK] Crop Detail test passed")

    def test_trees_catalog(self):
        response = self.client.get("/trees")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Teak", response.data)
        self.assertIn(b"Neem", response.data)
        self.assertIn(b"Harsingar", response.data)
        print("[OK] Trees Catalog test passed")

    def test_200_trees_dataset(self):
        from tree_data import all_trees, get_tree
        trees = all_trees()
        self.assertEqual(len(trees), 200)
        self.assertIsNotNone(get_tree("neem"))
        self.assertIsNotNone(get_tree("harsingar"))
        self.assertIsNotNone(get_tree("bilva"))
        self.assertIsNotNone(get_tree("baobab"))
        self.assertIsNotNone(get_tree("sandalwood"))
        print("[OK] Exactly 200 Trees verified in dataset (Neem to Harsingar)")

    def test_tree_categories(self):
        from tree_data import search_trees
        timber = search_trees(category="timber")
        fruit = search_trees(category="fruit")
        medicinal = search_trees(category="medicinal")
        sacred = search_trees(category="sacred")
        ornamental = search_trees(category="ornamental")
        self.assertGreater(len(timber), 0)
        self.assertGreater(len(fruit), 0)
        self.assertGreater(len(medicinal), 0)
        self.assertGreater(len(sacred), 0)
        self.assertGreater(len(ornamental), 0)
        print(f"[OK] Tree categories verified: {len(timber)} timber, {len(fruit)} fruit, {len(medicinal)} medicinal, {len(sacred)} sacred, {len(ornamental)} ornamental")

    def test_tree_detail(self):
        response = self.client.get("/tree/teak")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tectona grandis", response.data)
        
        resp_harsingar = self.client.get("/tree/harsingar")
        self.assertEqual(resp_harsingar.status_code, 200)
        self.assertIn(b"Nyctanthes arbor-tristis", resp_harsingar.data)
        print("[OK] Tree Detail test passed (Teak and Harsingar)")

    def test_leaf_scanner_with_sample(self):
        response = self.client.get("/leaf?sample=tomato_early_blight")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Tomato Early Blight", response.data)
        self.assertIn(b"Trichoderma", response.data)
        self.assertIn(b"Mancozeb", response.data)
        print("[OK] Leaf Disease Scanner test passed")

    def test_tree_scanner_with_sample(self):
        response = self.client.get("/tree?sample=neem")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Neem", response.data)
        self.assertIn(b"Azadirachta indica", response.data)
        print("[OK] Tree Recognition Scanner test passed")

    def test_camera_views(self):
        resp_leaf = self.client.get("/leaf?camera=1")
        self.assertEqual(resp_leaf.status_code, 200)
        self.assertIn(b"LIVE CAM", resp_leaf.data)
        self.assertIn(b"inlineCameraVideo", resp_leaf.data)
        self.assertIn(b"shutter-button", resp_leaf.data)

        resp_tree = self.client.get("/tree?camera=1")
        self.assertEqual(resp_tree.status_code, 200)
        self.assertIn(b"LIVE CAM", resp_tree.data)
        self.assertIn(b"inlineTreeCameraVideo", resp_tree.data)
        self.assertIn(b"shutter-button", resp_tree.data)
        print("[OK] Camera Viewfinder Tests passed for both Leaf and Tree scanners")

    def test_calculator_crop_post(self):
        response = self.client.post("/calculator", data={
            "mode": "crop",
            "crop_id": "rice",
            "acreage": "2.5",
            "yield_acre": "24",
            "price_quintal": "2200",
            "cost_acre": "18000"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Profitability Summary", response.data)
        self.assertIn(b"Gross Revenue", response.data)
        print("[OK] Crop Profitability Calculator test passed")

    def test_calculator_forestry_post(self):
        response = self.client.post("/calculator", data={
            "mode": "forestry",
            "tree_id": "teak",
            "dbh_cm": "50",
            "height_m": "15"
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Forestry", response.data)
        self.assertIn(b"Carbon", response.data)
        print("[OK] Forestry Timber & Carbon Calculator test passed")

    def test_multilingual_kannada_and_hindi(self):
        resp_kn = self.client.get("/?lang=kn")
        self.assertEqual(resp_kn.status_code, 200)
        self.assertIn("ಮುಖಪುಟ".encode('utf-8'), resp_kn.data)

        resp_hi = self.client.get("/?lang=hi")
        self.assertEqual(resp_hi.status_code, 200)
        self.assertIn("होम".encode('utf-8'), resp_hi.data)
        print("[OK] Multilingual Support (KN, HI) test passed")

    def test_about_page(self):
        response = self.client.get("/about")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"MobileNetV2", response.data)
        self.assertIn(b"Huber", response.data)
        print("[OK] About & Docs test passed")

    def test_200_crop_diseases_directory(self):
        from agri_ai import get_all_crop_diseases
        diseases = get_all_crop_diseases()
        self.assertEqual(len(diseases), 200, f"Expected 200 crop diseases, got {len(diseases)}")
        
        response = self.client.get("/crop-diseases")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"200 Crops Disease", response.data)
        self.assertIn(b"Organic", response.data)
        self.assertIn(b"Chemical", response.data)
        print(f"[OK] 200 Crops Disease Database verified ({len(diseases)} disease profiles)")

    def test_200_tree_images_exist(self):
        import os
        from tree_data import all_trees
        trees = all_trees()
        self.assertEqual(len(trees), 200)
        
        missing = []
        for t in trees:
            svg_path = os.path.join("static", "images", "trees", f"{t['id']}.svg")
            if not os.path.exists(svg_path):
                missing.append(svg_path)
        self.assertEqual(len(missing), 0, f"Missing {len(missing)} tree SVG images: {missing[:5]}")
        print(f"[OK] All 200 Botanical Tree Images verified in static/images/trees/")

    def test_jpg_file_upload(self):
        import io
        from PIL import Image
        img = Image.new("RGB", (120, 120), color=(46, 125, 50))
        img_io = io.BytesIO()
        img.save(img_io, format="JPEG")
        img_io.seek(0)
        
        response = self.client.post(
            "/leaf",
            data={"image": (img_io, "test_leaf.jpg"), "crop_id": "rice"},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"AI Diagnosis Report", response.data)
        self.assertIn(b"Rice", response.data)
        print("[OK] JPG File Upload & Disease Diagnosis test passed")

    def test_pdf_file_upload_and_extraction(self):
        import io
        from PIL import Image
        img = Image.new("RGB", (150, 150), color=(100, 180, 80))
        pdf_io = io.BytesIO()
        img.save(pdf_io, format="PDF")
        pdf_io.seek(0)
        
        response = self.client.post(
            "/leaf",
            data={"image": (pdf_io, "field_report_sample.pdf"), "crop_id": "tomato"},
            content_type="multipart/form-data"
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Extracted & Analyzed from Uploaded PDF Document", response.data)
        self.assertIn(b"AI Diagnosis Report", response.data)
        print("[OK] PDF File Upload & Document Extraction test passed")

if __name__ == "__main__":
    unittest.main()
