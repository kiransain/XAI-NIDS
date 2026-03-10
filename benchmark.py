import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
	accuracy_score,
	precision_recall_fscore_support,
	roc_auc_score,
	classification_report,
	confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping


warnings.filterwarnings("ignore")
RANDOM_STATE = 42
RESULTS_ROOT = os.path.join(os.getcwd(), "results_benchmark")
DATASETS = ["5G-NIDD", "5GAD", "5GC_PFCP"]
MODELS = ["DecisionTree", "RandomForest", "XGBoost", "MLP", "DNN"]


def _ensure_dirs(dataset: str) -> str:
	out_dir = os.path.join(RESULTS_ROOT, dataset)
	os.makedirs(out_dir, exist_ok=True)
	return out_dir


# ----------------------------
# 数据加载与预处理（独立实现）
# ----------------------------

def _preprocess_5g_nidd(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
	df = pd.read_csv(csv_path, encoding="utf-8")
	# 列维度清理（保留非NaN足够多的列）
	min_non_null = max(1000, int(0.7 * len(df)))
	df_cleaned = df.dropna(axis=1, thresh=min_non_null)

	# 合并少数类别
	if "Proto" in df_cleaned.columns:
		df_cleaned["Proto"] = np.where(
			df_cleaned["Proto"].isin(["lldp", "llc", "arp", "ipv6-icmp"]), "other", df_cleaned["Proto"]
		)
	if "State" in df_cleaned.columns:
		df_cleaned["State"] = np.where(
			df_cleaned["State"].isin(["ECO", "ACC", "URP", "RSP", "TST", "NRS"]), "other", df_cleaned["State"]
		)

	# one-hot编码
	cat_cols = [c for c in ["Proto", "Cause", "State"] if c in df_cleaned.columns]
	df_new = pd.get_dummies(df_cleaned, columns=cat_cols, prefix=cat_cols)

	# 删除明显非特征列
	drop_cols = [c for c in ["sDSb"] if c in df_new.columns]
	if drop_cols:
		df_new.drop(drop_cols, axis=1, inplace=True)
	# 行缺失清理
	df_new = df_new.dropna(axis=0)

	# Label / Attack Type 期望存在
	if "Label" not in df_new.columns and "label" in df_new.columns:
		df_new.rename(columns={"label": "Label"}, inplace=True)

	features = df_new.columns.tolist()
	for c in ["Label", "Attack Type", "Attack Tool", "Unnamed: 0"]:
		if c in features:
			features.remove(c)

	X = df_new[features]
	if "Label" in df_new.columns:
		binary_y = df_new["Label"].astype(str)
	else:
		raise ValueError("5G-NIDD 缺少 Label 列")

	return X, binary_y, df_new


def _preprocess_5gc_pfcp(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
	df = pd.read_csv(csv_path, encoding="utf-8")
	# 创建 Attack Type = 原始 Label，二分类 Label = Benign / Malicious
	if "Label" not in df.columns:
		raise ValueError("5GC_PFCP 缺少 Label 列")
	df_new = df.copy()
	df_new["Attack Type"] = df_new["Label"].astype(str)
	df_new["Label"] = np.where(df_new["Label"].astype(str) == "Normal", "Benign", "Malicious")

	features = df_new.columns.tolist()
	for c in ["Label", "Attack Type"]:
		if c in features:
			features.remove(c)
	X = df_new[features]
	binary_y = df_new["Label"].astype(str)
	return X, binary_y, df_new


def _preprocess_5gad(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
	df = pd.read_csv(csv_path, encoding="utf-8")
	# 删除IP列
	for c in ["src_ip", "dst_ip"]:
		if c in df.columns:
			df.drop(columns=[c], inplace=True)
	# 改名 attack_type -> Attack Type
	if "attack_type" in df.columns and "Attack Type" not in df.columns:
		df.rename(columns={"attack_type": "Attack Type"}, inplace=True)
	# tcp_flags 映射
	if "tcp_flags" in df.columns:
		mapping = {"0": 0, "A": 1, "PA": 2, "FPA": 3}
		df["tcp_flags"] = df["tcp_flags"].map(mapping).fillna(0)
	# Label 映射
	if "label" in df.columns:
		df["Label"] = df["label"].map({"normal": "Benign", "attack": "Malicious"})
		df.drop(columns=["label"], inplace=True)

	features = df.columns.tolist()
	for c in ["Label", "Attack Type"]:
		if c in features:
			features.remove(c)
	X = df[features]
	if "Label" not in df.columns:
		raise ValueError("5GAD 缺少 Label 列")
	binary_y = df["Label"].astype(str)
	return X, binary_y, df


def load_dataset(dataset: str) -> Tuple[pd.DataFrame, np.ndarray, LabelEncoder, pd.DataFrame]:
	cwd = os.getcwd()
	if dataset == "5G-NIDD":
		X, y_text, df_full = _preprocess_5g_nidd(os.path.join(cwd, "5G-NIDD.csv"))
	elif dataset == "5GC_PFCP":
		X, y_text, df_full = _preprocess_5gc_pfcp(os.path.join(cwd, "5GC_PFCP.csv"))
	elif dataset == "5GAD":
		X, y_text, df_full = _preprocess_5gad(os.path.join(cwd, "5GAD.csv"))
	else:
		raise ValueError(f"未知数据集: {dataset}")

	le = LabelEncoder()
	y = le.fit_transform(y_text)
	return X, y, le, df_full


def prepare_multiclass(df_full: pd.DataFrame, dataset: str) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
	if "Label" not in df_full.columns:
		raise ValueError("数据缺少 Label 列，无法提取恶意子集用于多分类")
	malicious = df_full[df_full["Label"].astype(str) == "Malicious"].copy()
	features = malicious.columns.tolist()
	for c in ["Label", "Attack Type"]:
		if c in features:
			features.remove(c)
	if dataset == "5G-NIDD":
		for c in ["Attack Tool", "Unnamed: 0"]:
			if c in features:
				features.remove(c)
	if not features:
		raise ValueError("多分类无可用特征列")
	if "Attack Type" not in malicious.columns:
		raise ValueError("多分类缺少 Attack Type 列")

	X_multi = malicious[features]
	le_multi = LabelEncoder()
	y_multi = le_multi.fit_transform(malicious["Attack Type"].astype(str))
	return X_multi, y_multi, le_multi.classes_.tolist()


# ----------------------------
# 模型构建（统一参数）
# ----------------------------

def make_dt() -> DecisionTreeClassifier:
	return DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced")


def make_rf() -> RandomForestClassifier:
	return RandomForestClassifier(
		n_estimators=100,
		max_depth=10,
		random_state=RANDOM_STATE,
		class_weight="balanced",
	)


def make_xgb(task: str, num_classes: int = 2) -> XGBClassifier:
	if task == "binary":
		return XGBClassifier(
			objective="binary:logistic",
			eval_metric="logloss",
			n_estimators=200,
			max_depth=6,
			learning_rate=0.1,
			subsample=0.8,
			colsample_bytree=0.8,
			random_state=RANDOM_STATE,
			use_label_encoder=False,
		)
	else:
		return XGBClassifier(
			objective="multi:softprob",
			num_class=num_classes,
			eval_metric="mlogloss",
			n_estimators=300,
			max_depth=6,
			learning_rate=0.1,
			subsample=0.8,
			colsample_bytree=0.8,
			random_state=RANDOM_STATE,
			use_label_encoder=False,
		)


def make_mlp(input_dim: int, task: str, num_classes: int = 2) -> Sequential:
	model = Sequential()
	model.add(Dense(128, activation="relu", input_shape=(input_dim,)))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(64, activation="relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	if task == "binary":
		model.add(Dense(1, activation="sigmoid"))
		model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
	else:
		model.add(Dense(num_classes, activation="softmax"))
		model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	return model


def make_dnn(input_dim: int, task: str, num_classes: int = 2) -> Sequential:
	model = Sequential()
	model.add(Dense(256, activation="relu", input_shape=(input_dim,)))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(128, activation="relu"))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Dense(64, activation="relu"))
	if task == "binary":
		model.add(Dense(1, activation="sigmoid"))
		model.compile(optimizer=Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
	else:
		model.add(Dense(num_classes, activation="softmax"))
		model.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
	return model


# ----------------------------
# 训练与评估
# ----------------------------

def evaluate_binary(X: pd.DataFrame, y: np.ndarray, dataset: str) -> pd.DataFrame:
	# 切分
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
	)

	# 标准化（仅用于神经网络）
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	# 类不平衡处理
	classes = np.unique(y_train)
	class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
	class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
	sample_weight = np.array([class_weight_dict[int(c)] for c in y_train])

	results = []

	for name in MODELS:
		if name == "DecisionTree":
			model = make_dt()
			model.fit(X_train, y_train)
			proba = model.predict_proba(X_test)[:, 1]
			pred = (proba >= 0.5).astype(int)
		elif name == "RandomForest":
			model = make_rf()
			model.fit(X_train, y_train)
			proba = model.predict_proba(X_test)[:, 1]
			pred = (proba >= 0.5).astype(int)
		elif name == "XGBoost":
			# 设置scale_pos_weight
			counts = np.bincount(y_train)
			model = make_xgb("binary")
			if len(counts) == 2 and counts[1] > 0:
				spw = counts[0] / counts[1]
				model.set_params(scale_pos_weight=float(spw))
			model.fit(X_train, y_train, sample_weight=sample_weight)
			proba = model.predict_proba(X_test)[:, 1]
			pred = (proba >= 0.5).astype(int)
		elif name == "MLP":
			model = make_mlp(X_train.shape[1], task="binary")
			es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
			model.fit(
				X_train_scaled,
				y_train,
				epochs=60,
				batch_size=512,
				validation_split=0.1,
				verbose=0,
				class_weight=class_weight_dict,
				callbacks=[es],
			)
			proba = model.predict(X_test_scaled, verbose=0).reshape(-1)
			pred = (proba >= 0.5).astype(int)
		elif name == "DNN":
			model = make_dnn(X_train.shape[1], task="binary")
			es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
			model.fit(
				X_train_scaled,
				y_train,
				epochs=60,
				batch_size=512,
				validation_split=0.1,
				verbose=0,
				class_weight=class_weight_dict,
				callbacks=[es],
			)
			proba = model.predict(X_test_scaled, verbose=0).reshape(-1)
			pred = (proba >= 0.5).astype(int)
		else:
			continue

		acc = accuracy_score(y_test, pred)
		prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, average="binary", zero_division=0)
		try:
			auc = roc_auc_score(y_test, proba)
		except Exception:
			auc = np.nan

		results.append(
			{
				"dataset": dataset,
				"task": "binary",
				"model": name,
				"accuracy": float(acc),
				"precision": float(prec),
				"recall": float(rec),
				"f1": float(f1),
				"auc": float(auc) if not np.isnan(auc) else np.nan,
			}
		)

	return pd.DataFrame(results)


def evaluate_multiclass(X: pd.DataFrame, y: np.ndarray, dataset: str, class_names: List[str]) -> pd.DataFrame:
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
	)

	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train)
	X_test_scaled = scaler.transform(X_test)

	classes = np.unique(y_train)
	class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
	class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_weights)}
	sample_weight = np.array([class_weight_dict[int(c)] for c in y_train])

	results = []

	for name in MODELS:
		if name == "DecisionTree":
			model = make_dt()
			model.fit(X_train, y_train)
			pred = model.predict(X_test)
			proba = model.predict_proba(X_test)
		elif name == "RandomForest":
			model = make_rf()
			model.fit(X_train, y_train)
			pred = model.predict(X_test)
			proba = model.predict_proba(X_test)
		elif name == "XGBoost":
			model = make_xgb("multi", num_classes=len(class_names))
			model.fit(X_train, y_train, sample_weight=sample_weight)
			pred = model.predict(X_test)
			proba = model.predict_proba(X_test)
		elif name == "MLP":
			model = make_mlp(X_train.shape[1], task="multi", num_classes=len(class_names))
			es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
			model.fit(
				X_train_scaled,
				y_train,
				epochs=60,
				batch_size=512,
				validation_split=0.1,
				verbose=0,
				class_weight=class_weight_dict,
				callbacks=[es],
			)
			proba = model.predict(X_test_scaled, verbose=0)
			pred = np.argmax(proba, axis=1)
		elif name == "DNN":
			model = make_dnn(X_train.shape[1], task="multi", num_classes=len(class_names))
			es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
			model.fit(
				X_train_scaled,
				y_train,
				epochs=60,
				batch_size=512,
				validation_split=0.1,
				verbose=0,
				class_weight=class_weight_dict,
				callbacks=[es],
			)
			proba = model.predict(X_test_scaled, verbose=0)
			pred = np.argmax(proba, axis=1)
		else:
			continue

		acc = accuracy_score(y_test, pred)
		prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
			y_test, pred, average="macro", zero_division=0
		)
		prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
			y_test, pred, average="weighted", zero_division=0
		)

		# 可选 AUC（OVR 宏平均），需要概率
		try:
			y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))
			auc_macro_ovr = roc_auc_score(y_test_bin, proba, average="macro", multi_class="ovr")
		except Exception:
			auc_macro_ovr = np.nan

		results.append(
			{
				"dataset": dataset,
				"task": "multiclass",
				"model": name,
				"accuracy": float(acc),
				"precision_macro": float(prec_macro),
				"recall_macro": float(rec_macro),
				"f1_macro": float(f1_macro),
				"f1_weighted": float(f1_weighted),
				"auc_macro_ovr": float(auc_macro_ovr) if not np.isnan(auc_macro_ovr) else np.nan,
			}
		)

	return pd.DataFrame(results)


def run_for_dataset(dataset: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
	print("\n" + "=" * 80)
	print(f"Dataset: {dataset}")
	print("=" * 80)
	out_dir = _ensure_dirs(dataset)

	X, y, le, df_full = load_dataset(dataset)
	binary_df = evaluate_binary(X, y, dataset)

	# 多分类：恶意子集 + Attack Type
	try:
		X_multi, y_multi, class_names = prepare_multiclass(df_full, dataset)
		multi_df = evaluate_multiclass(X_multi, y_multi, dataset, class_names)
	except Exception as e:
		print(f"[{dataset}] 多分类阶段跳过: {e}")
		multi_df = pd.DataFrame()

	# 保存
	if not binary_df.empty:
		binary_path = os.path.join(out_dir, "binary_metrics.csv")
		binary_df.to_csv(binary_path, index=False, encoding="utf-8-sig")
		print(f"Saved: {binary_path}")
	if not multi_df.empty:
		multi_path = os.path.join(out_dir, "multiclass_metrics.csv")
		multi_df.to_csv(multi_path, index=False, encoding="utf-8-sig")
		print(f"Saved: {multi_path}")

	# 控制台摘要
	if not binary_df.empty:
		print("\n[Binary] Summary:")
		print(binary_df[["model", "accuracy", "precision", "recall", "f1", "auc"]])
	if not multi_df.empty:
		print("\n[Multiclass] Summary:")
		print(multi_df[["model", "accuracy", "f1_macro", "f1_weighted", "auc_macro_ovr"]])

	return binary_df, multi_df


def main():
	os.makedirs(RESULTS_ROOT, exist_ok=True)
	all_bin, all_multi = [], []
	for ds in DATASETS:
		bin_df, multi_df = run_for_dataset(ds)
		if not bin_df.empty:
			all_bin.append(bin_df)
		if not multi_df.empty:
			all_multi.append(multi_df)

	if all_bin:
		pd.concat(all_bin, ignore_index=True).to_csv(
			os.path.join(RESULTS_ROOT, "binary_metrics_all.csv"), index=False, encoding="utf-8-sig"
		)
	if all_multi:
		pd.concat(all_multi, ignore_index=True).to_csv(
			os.path.join(RESULTS_ROOT, "multiclass_metrics_all.csv"), index=False, encoding="utf-8-sig"
		)


if __name__ == "__main__":
	main()
