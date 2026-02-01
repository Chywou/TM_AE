from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib

import numpy as np
from ctapipe.io import EventSource
from ctapipe.instrument.camera import CameraGeometry
from ctapipe.visualization import CameraDisplay
from ctapipe.image.cleaning import tailcuts_clean, apply_time_delta_cleaning
import matplotlib.pyplot as plt
from ctapipe.image import hillas_parameters, leakage_parameters, timing_parameters

import glob
import os
import argparse
import time
import hexagdly
import random
from datetime import datetime
from collections import deque
import random
import yaml
import pandas as pd
import h5py


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def clean_image_improvement(peak, mask, adj_list, max_diff=5):
    new_mask = mask.copy().astype(np.uint8)
    queue = deque(np.where(new_mask == 1)[0])

    while queue:
        i = queue.popleft()
        for n in adj_list[i]:
            if new_mask[n] == 0 and abs(peak[i] - peak[n]) <= max_diff:
                new_mask[n] = 1
                queue.append(n)

    return new_mask

proton_path = "/mnt_data/SST1M/data/protons_diffuse/reduce_train"

file_list = glob.glob(os.path.join(proton_path, "*.h5"))
with EventSource(file_list[0]) as source:
    for i, event in enumerate(source):
        for tel_id, tel_event in event.dl1.tel.items():
            geo = source.subarray.tel[tel_id].camera.geometry
        break

adj_list = geo.neighbors

def creat_x_y(x, y):
    """
    Create new x and y arrays with consecutive integer values.
    Mapping X and Y from telecope coordinates to 2D grid coordinates.
    """
    unique_vals = np.sort(np.unique(y))
    y_new = np.floor(len(unique_vals) / 2).astype(int)
    mapping = {val: i // 2 for i, val in enumerate(unique_vals)}
    y_new = np.array([mapping[val] for val in y])

    unique_vals = np.sort(np.unique(x))
    mapping = {val: i for i, val in enumerate(unique_vals)}
    x_new = np.array([mapping[val] for val in x])

    return x_new, y_new

x_new, y_new = creat_x_y(geo.pix_x.value, geo.pix_y.value)

H = len(np.unique(y_new))
W = len(np.unique(x_new)) + 1

def tensor_to_data(tensor, x_new, y_new, data_length=1296):
    data_reconstructed = np.zeros(data_length, dtype=np.float32)
    for i in range(data_length):
        data_reconstructed[i] = tensor[0, y_new[i], x_new[i]+1].item()
    return data_reconstructed

def data_to_tensor(img, x_new, y_new, H=36, W=37):
    img_tensor = torch.zeros((H, W), dtype=torch.float32)          
    img_tensor[y_new, x_new + 1] = torch.from_numpy(img.astype(np.float32))
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
    return img_tensor

def load_data(folder_path, max_events=None):
    telescopes = ["tel_001"]
    images_list = []
    masks_list = []
    peaks_list = []
    true_energy_list = []

    file_list = glob.glob(os.path.join(folder_path, "*.h5"))
    random.seed(0)
    random.shuffle(file_list)
    print(f"Found {len(file_list)} files", flush=True)

    for file_idx, file_path in enumerate(file_list, 1):
        with h5py.File(file_path, "r") as f:
            
            ds_energy = f["simulation/event/subarray/shower"][:]
            energy_by_event = {
                row["event_id"]: row["true_energy"]
                for row in ds_energy
            }

            for tel in telescopes:
                ds_image = f[f"dl1/event/telescope/images/{tel}"][:]
                ds_params = f[f"dl1/event/telescope/parameters/{tel}"][:]

                n_events = ds_image.shape[0]

                for i in range(n_events):

                    image = ds_image[i]["image"].astype(np.float32)
                    peak = ds_image[i]["peak_time"].astype(np.float32)

                    mask = tailcuts_clean(geo, image, picture_thresh=8, boundary_thresh=4, keep_isolated_pixels=False, min_number_picture_neighbors=2)
                    mask_modified = apply_time_delta_cleaning(geo, mask, peak, min_number_neighbors=1, time_limit=8)

                    if np.sum(mask_modified) <= 2:
                        continue

                    hillas = hillas_parameters(geo[mask_modified], image[mask_modified])
                    leakage = leakage_parameters(geo, image, mask_modified)

                    if hillas.intensity < 50 or hillas.intensity > 1000000:
                        continue

                    if leakage.intensity_width_2 > 0.7:
                        continue

                    event_id = ds_params[i]["event_id"]
                    true_energy = energy_by_event[event_id]

                    images_list.append(image)
                    peaks_list.append(peak)
                    masks_list.append(mask_modified)
                    true_energy_list.append(true_energy.astype(np.float32))

                    if max_events is not None and len(images_list) >= max_events:
                        return images_list, masks_list, peaks_list, true_energy_list

    return images_list, masks_list, peaks_list, true_energy_list

def hillas_to_rf_dataframe(hillas_list, particle_type, leakage_2, slopes):
    data = {
        "log_intensity": [],
        "width": [],
        "length": [],
        "wl": [],
        "skewness": [],
        "kurtosis": [],
        "leakage_2": leakage_2,
        "x": [],
        "y": [],
        "timing_slope": [],
        "type": [],
    }

    for hillas in hillas_list:

        width = hillas.width.value
        length = hillas.length.value

        wl = width / length if length > 0 else np.nan

        data["log_intensity"].append(np.log10(hillas.intensity) if hillas.intensity > 0 else 0)
        data["width"].append(width)
        data["length"].append(length)
        data["wl"].append(wl)
        data["skewness"].append(hillas.skewness)
        data["kurtosis"].append(hillas.kurtosis)

        data["x"].append(hillas.x.value)
        data["y"].append(hillas.y.value)
        data["type"].append(particle_type)

    for slope in slopes:
        data["timing_slope"].append(slope.value)


    return pd.DataFrame(data)

def get_data(path, ae, particle_type, max_events=None):
    images_list, masks_list, peaks_list, _ = load_data(path, max_events)

    hillas_params = []
    hillas_improved_params = []

    leakage_2 = []
    leakage_2_improved = []

    timing_slops = []
    timing_slops_improved = []

    ae.eval()

    for idx in range(len(images_list)):
        # print progress
        if idx % 5000 == 0:
            print(f"Processing event {idx}/{len(images_list)}", flush=True)
        image = images_list[idx]
        mask = masks_list[idx]
        peak = peaks_list[idx]

        image = np.clip(image, 0, None)
        image_hillas = image * mask
        hillas = hillas_parameters(geo[mask], image_hillas[mask])
        lk_2 = leakage_parameters(geo, image, mask).intensity_width_2
        ts = timing_parameters(geo, image, peak, hillas, mask).slope

        new_mask = clean_image_improvement(peak, mask, adj_list, max_diff=5).astype(bool)

        img = image * new_mask
        img = np.log1p(img)

        img_tensor = data_to_tensor(img, x_new, y_new, H, W)

        with torch.no_grad():
            reconstructed_tensor = ae(img_tensor.unsqueeze(0).to(device))

        reconstructed_img = tensor_to_data(reconstructed_tensor.squeeze(0), x_new, y_new)

        converted_back = np.expm1(reconstructed_img)
        converted_back[converted_back < 0] = 0

        hillas_rec = hillas_parameters(geo[new_mask], converted_back[new_mask])
        lk_2_rec = leakage_parameters(geo, converted_back, new_mask).intensity_width_2
        ts_rec = timing_parameters(geo, converted_back, peak, hillas_rec, new_mask).slope

        hillas_params.append(hillas)
        hillas_improved_params.append(hillas_rec)
        
        leakage_2.append(lk_2)
        leakage_2_improved.append(lk_2_rec)

        timing_slops.append(ts)
        timing_slops_improved.append(ts_rec)

    df = hillas_to_rf_dataframe(hillas_params, particle_type, leakage_2, timing_slops)
    df_improved = hillas_to_rf_dataframe(hillas_improved_params, particle_type, leakage_2_improved, timing_slops_improved)

    return df, df_improved

def get_classifier():
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=30,
        min_samples_leaf=10,
        min_samples_split=10,
        criterion="gini",
        bootstrap=True,
        oob_score=False,
        n_jobs=-1,
        random_state=42,
        verbose=0,
        warm_start=False,
        class_weight=None,
        min_weight_fraction_leaf=0.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
    )
    return rf

def get_dataset(df_gamma, df_proton):

    df = pd.concat([df_gamma, df_proton], ignore_index=True)

    X = df.drop(columns=["type"])
    y_type = df["type"]
    
    return X, y_type

class AE(nn.Module):
    def __init__(
        self,
        in_channels=1,
        enc1_out=8,
        enc2_out=4,
        latent_dim=32,
        k1=2,
        k2=1,
    ):
        super().__init__()

        # =========
        # Encoder
        # =========
        self.enc1 = hexagdly.Conv2d(
            in_channels=in_channels,
            out_channels=enc1_out,
            kernel_size=k1,
            stride=2,
            bias=True,
        )

        self.enc2 = hexagdly.Conv2d(
            in_channels=enc1_out,
            out_channels=enc2_out,
            kernel_size=k2,
            stride=2,
            bias=True,
        )

        self.flatten = nn.Flatten()
        self.fc_enc = nn.Linear(enc2_out * 9 * 13, latent_dim)

        # =========
        # Decoder
        # =========
        self.fc_dec = nn.Linear(latent_dim, enc2_out * 9 * 13)
        self.unflatten = nn.Unflatten(1, (enc2_out, 9, 13))

        self.dec1 = hexagdly.Conv2d(
            in_channels=enc2_out,
            out_channels=enc1_out,
            kernel_size=k2,
            stride=1,
            bias=True,
        )

        self.dec2 = hexagdly.Conv2d(
            in_channels=enc1_out,
            out_channels=in_channels,
            kernel_size=k1,
            stride=1,
            bias=True,
        )

    def forward(self, x):
        H, W = x.shape[-2:]

        # =========
        # Encoder
        # =========
        z = F.relu(self.enc1(x))          # (B, 8, 18, 25)
        z_first_shape = z.shape[-2:]
        z = F.relu(self.enc2(z))          # (B, 4, 9, 13)

        z = self.flatten(z)
        latent = self.fc_enc(z)           # (B, latent_dim)

        # =========
        # Decoder
        # =========
        z = self.fc_dec(latent)
        z = self.unflatten(z)

        z = F.interpolate(z, size=z_first_shape, mode="nearest")
        z = F.relu(self.dec1(z))

        z = F.interpolate(z, size=(H, W), mode="nearest")
        out = self.dec2(z)

        return out
    

    def load_model(self, path, device="cpu"):
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint['model_state_dict'])

def main(data_path_train_protons, data_path_train_gammas, data_path_test_protons, data_path_test_gammas, ae_model_path, model_save_dir, max_events=None, max_events_test=70000):
    print("Loading Autoencoder model...", flush=True)

    ae = AE()
    ae.load_model(ae_model_path, device=device)
    ae.eval()
    ae.to(device)

    print("Processing training data...", flush=True)

    df_proton, df_improved_proton = get_data(data_path_train_protons, ae, 0, max_events=max_events)
    df_gamma, df_improved_gamma = get_data(data_path_train_gammas, ae, 1, max_events=len(df_proton))
    df_proton_test, df_improved_proton_test = get_data(data_path_test_protons, ae, 0, max_events=max_events_test)
    df_gamma_test, df_improved_gamma_test = get_data(data_path_test_gammas, ae, 1, max_events=len(df_proton_test))
    
    print("Creating datasets...", flush=True)
    X, y = get_dataset(df_gamma, df_proton)
    X_improved, y_improved = get_dataset(df_improved_gamma, df_improved_proton)
    X_test, y_test = get_dataset(df_gamma_test, df_proton_test)
    X_improved_test, y_improved_test = get_dataset(df_improved_gamma_test, df_improved_proton_test)

    print("Training Random Forest Classifier...", flush=True)
    rf = get_classifier()
    rf.fit(X, y)

    print("Training Random Forest Classifier with improved data...", flush=True)
    rf_improved = get_classifier()
    rf_improved.fit(X_improved, y_improved)

    print("Testing Random Forest Classifier...", flush=True)
    test_score = rf.score(X_test, y_test)
    print(f"Test Accuracy (original data): {test_score:.4f}", flush=True)
    test_improved_score = rf_improved.score(X_improved_test, y_improved_test)
    print(f"Test Accuracy (improved data): {test_improved_score:.4f}", flush=True)

    # Save models
    rf_path = os.path.join(model_save_dir, "rf_classifier.pkl")
    rf_improved_path = os.path.join(model_save_dir, "rf_classifier_improved.pkl")

    joblib.dump(rf, rf_path)
    print(f"Saved Random Forest Classifier to {rf_path}", flush=True)
    joblib.dump(rf_improved, rf_improved_path)
    print(f"Saved Random Forest Classifier with improved data to {rf_improved_path}", flush=True)

    # Save results to CSV (predicted_probability, event type)
    results_df = pd.DataFrame({
        "predicted_proba_original": rf.predict_proba(X_test)[:, 1],
        "predicted_proba_improved": rf_improved.predict_proba(X_improved_test)[:, 1],
        "event_type": y_test
    })
    results_csv_path = os.path.join(model_save_dir, "rf_results_clf.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"Saved results to {results_csv_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train RF on telescope data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config["model_save_dir"]):
        print(f"Creating directory: {config['model_save_dir']}")
        os.makedirs(config["model_save_dir"], exist_ok=True)

    main(
        data_path_train_protons=config["path_train_protons"],
        data_path_train_gammas=config["path_train_gammas"],
        data_path_test_protons=config["path_test_protons"],
        data_path_test_gammas=config["path_test_gammas"],
        ae_model_path=config["ae_model_path"],
        model_save_dir=config["model_save_dir"],
        max_events=config.get("max_events", None),
        max_events_test=config.get("max_events_test", 70000)
    )
