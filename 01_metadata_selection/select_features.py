#!/usr/bin/env python3
"""Select stable metadata features for binary or multiclass domain classification."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
STEP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = (
    STEP_DIR.parent / "09_output" / STEP_DIR.name / "threatfox_feature_selection"
)


class MetadataFeatureSelector:
    NON_FEATURE_COLUMNS = {"input", "class", "label"}
    IGNORED_COLUMNS = {"_error", "_elapsed_s", "label_int"}
    METRICS = ("macro_f1", "balanced_accuracy", "mcc", "auroc", "ap")

    def __init__(
        self,
        missing_threshold=0.95,
        dominance_threshold=0.995,
        min_mi=0.001,
        validation_tolerance=0.005,
        n_splits=5,
        stability_threshold=0.80,
        random_state=42,
        dataset_name="ThreatFox Malware",
        target_column="label",
    ):
        if not 0 < missing_threshold <= 1:
            raise ValueError("missing_threshold must be in (0, 1]")
        if not 0 < dominance_threshold <= 1:
            raise ValueError("dominance_threshold must be in (0, 1]")
        if min_mi < 0:
            raise ValueError("min_mi must be non-negative")
        if validation_tolerance < 0:
            raise ValueError("validation_tolerance must be non-negative")
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not 0 < stability_threshold <= 1:
            raise ValueError("stability_threshold must be in (0, 1]")
        if target_column not in {"label", "class"}:
            raise ValueError("target_column must be 'label' or 'class'")
        self.missing_threshold = missing_threshold
        self.dominance_threshold = dominance_threshold
        self.min_mi = min_mi
        self.validation_tolerance = validation_tolerance
        self.n_splits = n_splits
        self.stability_threshold = stability_threshold
        self.random_state = random_state
        self.dataset_name = dataset_name
        self.target_column = target_column

    # Step 1: load train/dev and isolate the configured target and metadata columns.
    def preprocess_training_data(self, train_csv, dev_csv):
        self.train = pd.read_csv(train_csv)
        self.dev = pd.read_csv(dev_csv)
        self._validate_split(self.train, "Training")
        ignore = self.NON_FEATURE_COLUMNS | self.IGNORED_COLUMNS
        self.feature_columns = [
            column for column in self.train.columns if column not in ignore
        ]
        self._validate_split(self.dev, "Validation")
        self.X_train = self.train[self.feature_columns].apply(pd.to_numeric, errors="raise")
        self.X_dev = self.dev[self.feature_columns].apply(pd.to_numeric, errors="raise")

        train_targets = self.train[self.target_column].astype(str)
        dev_targets = self.dev[self.target_column].astype(str)
        train_classes = set(train_targets.unique())
        dev_classes = set(dev_targets.unique())
        if train_classes != dev_classes or len(train_classes) < 2:
            raise ValueError("Training and validation must contain the same target classes")
        if self.target_column == "label" and train_classes != {"legit", "malicious"}:
            raise ValueError("Binary label target must contain legit and malicious")

        class_counts = train_targets.value_counts()
        if class_counts.min() < self.n_splits:
            raise ValueError(
                f"Every target class needs at least {self.n_splits} training samples"
            )
        self.target_encoder = LabelEncoder().fit(train_targets)
        self.class_names = self.target_encoder.classes_.tolist()
        self.y_train = pd.Series(
            self.target_encoder.transform(train_targets), index=self.train.index
        )
        self.y_dev = pd.Series(
            self.target_encoder.transform(dev_targets), index=self.dev.index
        )

    def _validate_split(self, frame, name):
        required = {"input", self.target_column}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} split lacks columns: {sorted(missing)}")
        if frame[self.target_column].isna().any():
            raise ValueError(f"{name} split contains missing target values")
        if hasattr(self, "feature_columns"):
            missing = set(self.feature_columns) - set(frame.columns)
            if missing:
                raise ValueError(f"{name} split lacks features: {sorted(missing)}")

    # Step 2: remove constant, highly missing, and strongly dominated features.
    def filter_statistical_quality(self, X):
        missing = X.isna().mean()
        dominant = X.apply(
            lambda column: column.value_counts(dropna=True, normalize=True).max()
        )
        unique = X.nunique(dropna=True)
        reason = pd.Series("kept", index=self.feature_columns, dtype="object")
        reason[unique <= 1] = "constant"
        reason[(reason == "kept") & (missing >= self.missing_threshold)] = "high_missingness"
        reason[(reason == "kept") & (dominant >= self.dominance_threshold)] = "dominant_value"
        features = reason[reason == "kept"].index.tolist()
        if not features:
            raise ValueError("Statistical filtering removed every feature")
        values = SimpleImputer(strategy="median").fit_transform(X[features])
        clean = pd.DataFrame(values, columns=features, index=X.index)
        audit = pd.DataFrame(
            {
                "missing_fraction": missing,
                "dominant_fraction": dominant,
                "unique_non_missing": unique,
                "quality_status": reason,
            }
        )
        return features, clean, audit

    # Step 3: retain features with measurable mutual information with the label.
    def assess_predictive_relevance(self, X, y, features, audit):
        discrete = np.array([X[column].nunique() <= 20 for column in features])
        scores = mutual_info_classif(
            X,
            y,
            discrete_features=discrete,
            random_state=self.random_state,
        )
        mi = pd.Series(scores, index=features, name="mutual_information")
        relevant = mi[mi >= self.min_mi].index.tolist() or [mi.idxmax()]
        audit["mutual_information"] = mi
        audit["relevant"] = audit.index.isin(relevant)
        return relevant, mi


    def _select_once(self, X, y):
        quality, clean, audit = self.filter_statistical_quality(X)
        relevant, _ = self.assess_predictive_relevance(clean, y, quality, audit)
        audit["single_run_selected"] = audit.index.isin(relevant)
        return relevant, audit

    def _evaluate(self, X_fit, y_fit, X_eval, y_eval, features):
        is_multiclass = len(self.class_names) > 2
        classifier = LogisticRegression(
            class_weight="balanced",
            solver="lbfgs",
            max_iter=50 if is_multiclass else 1000,
            tol=1e-3 if is_multiclass else 1e-4,
            random_state=self.random_state,
            n_jobs=-1,
        )
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            classifier,
        )
        model.fit(X_fit[features], y_fit)
        prediction = model.predict(X_eval[features])
        probability = model.predict_proba(X_eval[features])
        if not is_multiclass:
            auroc = roc_auc_score(y_eval, probability[:, 1])
            ap = average_precision_score(y_eval, probability[:, 1])
        else:
            classes = np.arange(len(self.class_names))
            binary_targets = label_binarize(y_eval, classes=classes)
            auroc = roc_auc_score(
                y_eval,
                probability,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
            ap = average_precision_score(binary_targets, probability, average="macro")
        return {
            "macro_f1": f1_score(y_eval, prediction, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y_eval, prediction),
            "mcc": matthews_corrcoef(y_eval, prediction),
            "auroc": auroc,
            "ap": ap,
        }

    # Step 4: select within five folds, then compare all features with the stable subset.
    def assess_selection_stability(self):
        splitter = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True, random_state=self.random_state
        )
        folds = list(splitter.split(self.X_train, self.y_train))
        counts = pd.Series(0, index=self.feature_columns, dtype="int64")
        for fit_index, _ in folds:
            selected, _ = self._select_once(
                self.X_train.iloc[fit_index], self.y_train.iloc[fit_index]
            )
            counts[selected] += 1

        minimum_count = int(np.ceil(self.n_splits * self.stability_threshold))
        self.stable_features = [
            feature for feature in self.feature_columns if counts[feature] >= minimum_count
        ]
        if not self.stable_features:
            raise ValueError("No feature met the stability threshold")
        self.stability = pd.DataFrame(
            {
                "selection_count": counts,
                "selection_frequency": counts / self.n_splits,
                "stable": counts >= minimum_count,
            }
        )

        is_multiclass = len(self.class_names) > 2
        if not is_multiclass:
            rows = []
            for fold, (fit_index, eval_index) in enumerate(folds, start=1):
                X_fit = self.X_train.iloc[fit_index]
                y_fit = self.y_train.iloc[fit_index]
                X_eval = self.X_train.iloc[eval_index]
                y_eval = self.y_train.iloc[eval_index]
                for configuration, features in (
                    ("all_features", self.feature_columns),
                    ("stable_subset", self.stable_features),
                ):
                    rows.append(
                        {
                            "fold": fold,
                            "configuration": configuration,
                            "feature_count": len(features),
                            **self._evaluate(X_fit, y_fit, X_eval, y_eval, features),
                        }
                    )
            self.cv_metrics = pd.DataFrame(rows)
            aggregations = {"feature_count": ("feature_count", "first")}
            for metric in self.METRICS:
                aggregations[f"{metric}_mean"] = (metric, "mean")
                aggregations[f"{metric}_std"] = (metric, "std")
            self.cv_summary = (
                self.cv_metrics.groupby("configuration").agg(**aggregations).reset_index()
            )
        else:
            # ponytail: multiclass skips 10-fold linear fits; upgrade when fast probe is required.
            self.cv_summary = pd.DataFrame(
                [
                    {"configuration": "all_features", "feature_count": len(self.feature_columns)},
                    {"configuration": "stable_subset", "feature_count": len(self.stable_features)},
                ]
            )
        _, self.audit = self._select_once(self.X_train, self.y_train)
        self.audit = self.audit.join(self.stability)

    # Step 5: use dev macro-F1 once to freeze the stable subset or all features.
    def validate_and_freeze_selection(self):
        is_multiclass = len(self.class_names) > 2
        if not is_multiclass:
            rows = []
            for configuration, features in (
                ("all_features", self.feature_columns),
                ("stable_subset", self.stable_features),
            ):
                rows.append(
                    {
                        "configuration": configuration,
                        "feature_count": len(features),
                        **self._evaluate(
                            self.X_train, self.y_train, self.X_dev, self.y_dev, features
                        ),
                    }
                )
            self.validation_metrics = pd.DataFrame(rows)
            all_score, stable_score = self.validation_metrics["macro_f1"]
            self.selection_accepted = stable_score >= all_score - self.validation_tolerance
            self.selected_features = (
                self.stable_features if self.selection_accepted else self.feature_columns
            )
            self.decision = "stable_subset" if self.selection_accepted else "all_features"
            difference = stable_score - all_score
            if self.selection_accepted:
                self.justification = (
                    f"The stable subset was retained because its validation macro-F1 difference "
                    f"({difference:.6f}) was within the prespecified absolute margin "
                    f"({self.validation_tolerance:.3f})."
                )
            else:
                self.justification = (
                    f"All features were retained because the stable subset's validation macro-F1 "
                    f"difference ({difference:.6f}) exceeded the prespecified absolute margin "
                    f"({self.validation_tolerance:.3f})."
                )
        else:
            # 4-step protocol: freeze 5-fold stratified MI recurrence stable subset directly.
            self.selection_accepted = True
            self.selected_features = self.stable_features
            self.decision = "stable_subset"
            self.validation_metrics = pd.DataFrame(
                [
                    {"configuration": "all_features", "feature_count": len(self.feature_columns)},
                    {"configuration": "stable_subset", "feature_count": len(self.stable_features)},
                ]
            )
            self.justification = (
                f"For multiclass target ({len(self.class_names)} classes), the 4-step selection "
                f"pipeline retained the stable subset ({len(self.stable_features)} features) directly "
                f"from 5-fold stratified recurrence (>=80%), omitting the linear probe validation step."
            )
        self.audit["selected_final"] = self.audit.index.isin(self.selected_features)

    def _save_results(self, output_dir):
        (output_dir / "selected_features.txt").write_text(
            "\n".join(self.selected_features) + "\n"
        )
        self.audit.rename_axis("feature").to_csv(output_dir / "selection_audit.csv")
        self.stability.rename_axis("feature").to_csv(
            output_dir / "selection_stability.csv"
        )
        self.cv_summary.to_csv(output_dir / "cv_summary.csv", index=False)

        summary = {
            "dataset": self.dataset_name,
            "target": self.target_column,
            "class_count": len(self.class_names),
            "class_names": self.class_names,
            "primary_metric": "macro_f1",
            "pipeline_steps": 4 if len(self.class_names) > 2 else 5,
            "linear_validation_used": len(self.class_names) == 2,
            "classifier": "none (4-step filter/stability selection)" if len(self.class_names) > 2 else "median imputation + standard scaling + class-balanced logistic regression",
            "cross_validation": {
                "type": "StratifiedKFold",
                "folds": self.n_splits,
                "shuffle": True,
                "random_state": self.random_state,
            },
            "thresholds": {
                "missing_fraction": self.missing_threshold,
                "dominant_fraction": self.dominance_threshold,
                "minimum_mutual_information": self.min_mi,
                "stability_frequency": self.stability_threshold,
                "validation_macro_f1_margin": self.validation_tolerance,
            },
            "feature_counts": {
                "original": len(self.feature_columns),
                "stable_subset": len(self.stable_features),
                "final": len(self.selected_features),
            },
            "cv_summary": self.cv_summary.set_index("configuration").to_dict("index"),
            "validation_metrics": self.validation_metrics.set_index("configuration").to_dict("index"),
            "chosen_subset": self.decision,
            "justification": self.justification,
            "test_set_used_for_finalization": False,
        }
        (output_dir / "selection_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )

    def run(self, dataset_dir, output_dir):
        dataset_dir = Path(dataset_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.preprocess_training_data(dataset_dir / "train.csv", dataset_dir / "dev.csv")
        self.assess_selection_stability()
        self.validate_and_freeze_selection()
        self._save_results(output_dir)

        scores = self.cv_summary.set_index("configuration")
        print(f"Final selected feature count: {len(self.selected_features)}")
        if len(self.class_names) == 2:
            print(
                "CV macro-F1: "
                f"all {len(self.feature_columns)} = "
                f"{scores.at['all_features', 'macro_f1_mean']:.6f} ± "
                f"{scores.at['all_features', 'macro_f1_std']:.6f}; "
                f"stable {len(self.stable_features)} = "
                f"{scores.at['stable_subset', 'macro_f1_mean']:.6f} ± "
                f"{scores.at['stable_subset', 'macro_f1_std']:.6f}"
            )
        print(f"Decision: {self.decision}. {self.justification}")
        assert set(self.selected_features).issubset(self.feature_columns)
        return self.selected_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="Dataset directory containing split CSVs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="ThreatFox Malware")
    parser.add_argument("--target-column", choices=("label", "class"), default="label")
    parser.add_argument("--missing-threshold", type=float, default=0.95)
    parser.add_argument("--dominance-threshold", type=float, default=0.995)
    parser.add_argument("--min-mi", type=float, default=0.001)
    parser.add_argument("--validation-tolerance", type=float, default=0.005)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--stability-threshold", type=float, default=0.80)
    args = parser.parse_args()

    selector = MetadataFeatureSelector(
        missing_threshold=args.missing_threshold,
        dominance_threshold=args.dominance_threshold,
        min_mi=args.min_mi,
        validation_tolerance=args.validation_tolerance,
        n_splits=args.folds,
        stability_threshold=args.stability_threshold,
        dataset_name=args.dataset_name,
        target_column=args.target_column,
    )
    selector.run(args.dataset_dir, args.output_dir)


if __name__ == "__main__":
    main()
