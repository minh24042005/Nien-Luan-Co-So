"""Logic tests use tiny synthetic fixtures shaped like the genuine dataset files."""
import io
import unittest
import warnings

import numpy as np
import pandas as pd

from dataset_loader import NSL_KDD_FEATURES, read_cicids2017, read_nsl_kdd
from model_engine import SOCModelEngine
from preprocessor import NetworkDataPreprocessor, generate_sample_nslkdd_data


class LoaderTests(unittest.TestCase):
    def make_nsl(self, attack):
        row = [1] * len(NSL_KDD_FEATURES)
        for col, value in {"protocol_type": "tcp", "service": "http", "flag": "SF"}.items():
            row[NSL_KDD_FEATURES.index(col)] = value
        return ",".join(map(str, row + [attack, 21]))

    def test_nsl_43_columns_drop_difficulty_and_map_attack(self):
        data = "\n".join([self.make_nsl(x) for x in
                          ("normal", "neptune", "satan", "worm", "httptunnel", "sqlattack")])
        result = read_nsl_kdd(io.StringIO(data))
        self.assertEqual(len(result.columns), 42)
        self.assertNotIn("difficulty", result.columns)
        self.assertNotIn("attack_name", result.columns)
        self.assertEqual(result.label.tolist(),
                         ["Normal", "DoS", "Probe", "R2L", "U2R", "U2R"])

    def test_nsl_reject_wrong_schema(self):
        with self.assertRaisesRegex(ValueError, "42 hoặc 43"):
            read_nsl_kdd(io.StringIO("1,2,3,bad\n"))

    def test_nsl_reject_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "chưa được ánh xạ"):
            read_nsl_kdd(io.StringIO(self.make_nsl("not_an_attack")))

    def test_cicids_preserves_labels_and_discards_identifiers(self):
        text1 = "Flow ID, Source IP, Timestamp, Flow Duration, Total Fwd Packets, Flow Bytes/s, Label\n" \
                "a,192.168.0.1,123,10,2,Infinity,BENIGN\n" \
                "b,192.168.0.2,124,20,3,33,DoS Hulk\n"
        text2 = "Flow ID, Source IP, Timestamp, Flow Duration, Total Fwd Packets, Flow Bytes/s, Label\n" \
                "c,192.168.0.3,125,30,2,24,PortScan\n" \
                "d,192.168.0.4,126,40,2,-Infinity,BENIGN\n"
        frames = [io.StringIO(text1), io.StringIO(text2)]
        result = read_cicids2017(frames, max_rows_per_file=100)
        self.assertEqual(len(result), 4)
        self.assertIn("Normal", result.label.tolist())
        self.assertIn("DoS Hulk", result.label.tolist())
        self.assertIn("PortScan", result.label.tolist())
        self.assertNotIn("Source IP", result.columns)
        self.assertNotIn("Timestamp", result.columns)
        self.assertIn("Flow Duration", result.columns)

    def test_cicids_sampling_limit_per_file(self):
        rows = ["Flow Duration,Packets,Label"]
        rows += [f"{i},{i%4},{'BENIGN' if i%3 else 'DoS Hulk'}" for i in range(250)]
        data = read_cicids2017([io.StringIO("\n".join(rows))], max_rows_per_file=27)
        self.assertEqual(len(data), 27)


class PreprocessorTests(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({
            "duration": [0., 10., np.nan, 4., 5., 8.],
            "protocol_type": ["tcp", "tcp", "udp", "tcp", "udp", "tcp"],
            "service": ["http", "ftp", "http", "ftp", "http", "ftp"],
        })
        self.y = ["Normal", "DoS", "Normal", "DoS", "Normal", "DoS"]

    def test_fit_only_train_and_handle_unseen_category(self):
        prep = NetworkDataPreprocessor("MinMaxScaler", top_k=None)
        train = prep.fit_transform(self.X, self.y)
        incoming = pd.DataFrame({"duration": [100., np.nan],
                                 "protocol_type": ["icmp", "tcp"],
                                 "service": ["unknown", "http"]})
        test = prep.transform(incoming)
        self.assertListEqual(list(train.columns), list(test.columns))
        # Train max 10, so 100 on Test must not be re-fitted/clipped to 1.
        self.assertGreater(test["num__duration"].iloc[0], 1)
        # Train median from [0,10,4,5,8] is 5, not 0.
        self.assertAlmostEqual(test["num__duration"].iloc[1], 0.5)
        self.assertEqual(test.filter(like="cat__protocol_type_").iloc[0].sum(), 0)
        self.assertFalse(test.isna().any().any())

    def test_selection_retains_k_features(self):
        prep = NetworkDataPreprocessor("StandardScaler", top_k=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            train = prep.fit_transform(self.X, self.y)
        self.assertEqual(train.shape[1], 2)
        self.assertEqual(prep.transform(self.X).shape[1], 2)

    def test_simulation_end_to_end_train_test(self):
        demo = generate_sample_nslkdd_data(350)
        train_df = demo.iloc[:280].copy()
        test_df = demo.iloc[280:].copy()
        prep = NetworkDataPreprocessor("MinMaxScaler", top_k=10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            Xtrain = prep.fit_transform(train_df.drop(columns="label"), train_df.label)
        Xtest = prep.transform(test_df.drop(columns="label"))
        results = SOCModelEngine().train_and_evaluate(Xtrain, train_df.label,
                                                      Xtest, test_df.label)
        self.assertEqual(len(results), 3)
        self.assertTrue(all("Accuracy (%)" in metrics for metrics in results.values()))


if __name__ == "__main__":
    unittest.main()
