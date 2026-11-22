"""Testes de integridade do preparador v1. python -m unittest discover -s tests -p 'test_prepare_pilot_v1.py'"""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_pilot_v1.py"
spec = importlib.util.spec_from_file_location("prepare_pilot_v1", SCRIPT)
import sys
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


class TestPreparePilotV1(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for source, clips in {
            "youtube": [("hk_court1_trecho", 121)],
            "local": [("p1", 61), ("p2", 59)],
            "synthetic": [("BocciaVision_01", 72), ("BocciaVision_03", 49)],
        }.items():
            base = self.root / "data" / "roboflow" / source / "v1"
            (base / "images").mkdir(parents=True)
            (base / "labels").mkdir(parents=True)
            names = ["blue_ball", "red_ball", "white_ball"]
            if source == "local":
                names = ["red_ball", "white_ball", "blue_ball"]
            (base / "data.yaml").write_text(f"nc: 3\nnames: {json.dumps(names)}\n", encoding="utf-8")
            for clip, count in clips:
                for i in range(count):
                    stem = f"{clip}_mp4-{i:04d}_jpg.rf.{i:032x}"
                    (base / "images" / (stem + ".jpg")).write_bytes(f"unique image: {source}/{clip}/{i}".encode())
                    label = "" if i % 17 == 0 else "0 0.5 0.5 0.1 0.1\n"
                    (base / "labels" / (stem + ".txt")).write_text(label, encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def args(self, **kw):
        values = dict(root=self.root, version="v1", class_mode="color", budget=60,
                      seed=42, val_fraction=0.2, gap=5, output=None, dry_run=False)
        values.update(kw)
        return SimpleNamespace(**values)

    def test_dry_run_no_file_created(self):
        output = module.run(self.args(dry_run=True))
        self.assertFalse(output.exists())

    def test_six_datasets_no_leak_and_equal_budget(self):
        output = module.run(self.args())
        manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["total_images_input"], 362)
        for fold_name, test_group, n_test in (("rodada_a", "real_youtube_match_01", 121),
                                              ("rodada_b", "real_local_match_01", 120)):
            fold = manifest["folds"][fold_name]
            self.assertEqual(fold["external_test_group"], test_group)
            paths_of_valid = None
            paths_of_test = None
            for scenario, expected_source_counts in (("real", {"local": 60} if fold_name == "rodada_a" else {"youtube": 60}),
                                                     ("synthetic", {"synthetic": 60}),
                                                     ("hybrid", {"synthetic": 30, "local": 30} if fold_name == "rodada_a" else {"synthetic": 30, "youtube": 30})):
                data = fold["scenarios"][scenario]
                self.assertEqual(data["counts"]["train"], 60)
                self.assertEqual(data["counts"]["test"], n_test)
                self.assertEqual(data["train_by_source"], expected_source_counts)
                ids = {key: {item["image"] for item in data[key]} for key in ("train", "valid", "test")}
                self.assertFalse(ids["train"] & ids["valid"])
                self.assertFalse(ids["train"] & ids["test"])
                self.assertFalse(ids["valid"] & ids["test"])
                self.assertEqual({item["group_id"] for item in data["test"]}, {test_group})
                self.assertNotIn(test_group, {item["group_id"] for item in data["train"]})
                if paths_of_test is not None:
                    self.assertEqual(ids["test"], paths_of_test)
                    self.assertEqual(ids["valid"], paths_of_valid)
                paths_of_test = ids["test"]
                paths_of_valid = ids["valid"]
                yaml_data = (output / fold_name / scenario / "data.yaml").read_text(encoding="utf-8")
                self.assertIn(str((output / fold_name / scenario).resolve().as_posix()), yaml_data)
                images = list((output / fold_name / scenario / "train" / "images").glob("*.jpg"))
                self.assertEqual(len(images), 60)
        # Reexecutar nao sobrescreve nenhum resultado existente.
        with self.assertRaises(module.PreparationError):
            module.run(self.args())

    def test_color_class_remap_by_semantic_name(self):
        output = module.run(self.args())
        # Local tem class 0 = red_ball; ordem canonica exige class 1 = red_ball.
        label_dir = output / "rodada_a" / "real" / "train" / "labels"
        labels = [p for p in label_dir.glob("*.txt") if p.read_text()]
        self.assertTrue(labels)
        self.assertTrue(all(p.read_text().startswith("1 ") for p in labels))

    def test_class_remap_and_generic(self):
        output = module.run(self.args(class_mode="generic"))
        manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["class_names"], ["ball"])
        label_dir = output / "rodada_a" / "real" / "train" / "labels"
        populated = [p for p in label_dir.glob("*.txt") if p.read_text()]
        self.assertTrue(populated)
        self.assertTrue(all(p.read_text().startswith("0 ") for p in populated))

    def test_invalid_label_detected(self):
        label = next((self.root / "data" / "roboflow" / "youtube" / "v1" / "labels").glob("*.txt"))
        label.write_text("0 1.2 0.5 0.3 0.3", encoding="utf-8")
        with self.assertRaises(module.PreparationError):
            module.run(self.args(dry_run=True))

    def test_unknown_scene_fails_closed(self):
        base = self.root / "data" / "roboflow" / "synthetic" / "v1"
        (base / "images" / "new_scene_mp4-0000_jpg.rf.aaa.jpg").write_bytes(b"x")
        (base / "labels" / "new_scene_mp4-0000_jpg.rf.aaa.txt").write_text("", encoding="utf-8")
        with self.assertRaises(module.PreparationError):
            module.run(self.args(dry_run=True))


if __name__ == "__main__":
    unittest.main()
