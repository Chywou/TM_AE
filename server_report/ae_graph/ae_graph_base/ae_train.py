import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import numpy as np
from ctapipe.io import EventSource
from ctapipe.instrument.camera import CameraGeometry
import glob
import os
import argparse
import time
import random
from datetime import datetime
import random
import yaml
import pandas as pd
import h5py

# GPU check
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

proton_path = "/mnt_data/SST1M/data/protons_diffuse/reduce_train"

file_list = glob.glob(os.path.join(proton_path, "*.h5"))
with EventSource(file_list[0]) as source:
    for i, event in enumerate(source):
        for tel_id, tel_event in event.dl1.tel.items():
            geo = source.subarray.tel[tel_id].camera.geometry
        break
neighbors = geo.neighbors

adj_list = geo.neighbors
edge_index_list = [
    [i, j]
    for i, neighbors in enumerate(adj_list)
    for j in neighbors
]

edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous().to(device)

def load_data(folder_path, max_events=None):
    telescopes = ["tel_001", "tel_002"]
    images_list = []
    masks_list = []
    peaks_list = []
    true_energy_list = []

    file_list = glob.glob(os.path.join(folder_path, "*.h5"))
    random.seed(0)
    random.shuffle(file_list)
    print(f"Found {len(file_list)} files", flush=True)

    for file_idx, file_path in enumerate(file_list, 1):
        print(f"Processing file {file_idx}/{len(file_list)}", flush=True)
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

                    event_id = ds_params[i]["event_id"]
                    true_energy = energy_by_event[event_id]

                    images_list.append(ds_image[i]["image"].astype(np.float32))
                    peaks_list.append(ds_image[i]["peak_time"].astype(np.float32))
                    masks_list.append(ds_image[i]["image_mask"].astype(bool))
                    true_energy_list.append(true_energy.astype(np.float32))

                    if max_events is not None and len(images_list) >= max_events:
                        return images_list, masks_list, peaks_list, true_energy_list

    return images_list, masks_list, peaks_list, true_energy_list

class TelescopeDataset(Dataset):
    def __init__(self, folder_path, edge_index, max_events=None):

        self.folder_path = folder_path
        self.graphs = []
        self.edge_index = edge_index
        self.originals = []
        self.true_energies = []

        start_total = time.time()
        images, masks, peaks, true_energies = load_data(folder_path, max_events=max_events)
        self.true_energies = true_energies
        print(f"Loaded {len(images)} events.", flush=True)
        print("Applying preprocessing...", flush=True)

        self._build_dataset(images, masks, peaks)

        end_total = time.time()
        print(f"Total loading time: {end_total - start_total:.2f} seconds", flush=True)
        print(f"Total events loaded: {len(self.graphs)}", flush=True)

    def _build_dataset(self, images, masks, peaks):
        for image, msk, pk in zip(images, masks, peaks):

            # Preprocess the image
            img = np.clip(image, 0, None)
            img = img / (np.max(img) + 1e-8)

            x = torch.from_numpy(img).float().unsqueeze(1)

            # Graph Data
            data = Data(
                x=x,
                edge_index=self.edge_index
            )

            self.graphs.append(data)

            self.originals.append(image)
    
    def get_true_energies(self):
        return self.true_energies

    def get_originals(self):
        return self.originals
                                                          
    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]
    
class AE(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=32, latent_dim=4):
        super().__init__()
        # Encodeur GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)
        
        # Décodeur MLP
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, in_channels)
        )

    def encode(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        z = self.conv2(x, edge_index)
        return z

    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        x_hat = self.decode(z)

        return x_hat

    
def reconstruct_dataset(model, dataset, batch_size=16):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_nodes = 1296
    
    originals = []
    normalized = []
    reconstructed = []

    with torch.no_grad():
        for batch in dataloader:   
            batch = batch.to(device)

            x_hat = model(batch.x, batch.edge_index)

            batch_size_actual = batch.num_graphs
            images = batch.x.view(batch_size_actual, num_nodes, -1).cpu()
            outputs = x_hat.view(batch_size_actual, num_nodes, -1).cpu()

            normalized.append(images)
            reconstructed.append(outputs)

    originals = dataset.get_originals()
    normalized = torch.cat(normalized, dim=0)
    reconstructed = torch.cat(reconstructed, dim=0)

    return originals, normalized, reconstructed

def reconstruction_error(original, reconstructed, criterion):
    print(f"Shape original: {original.shape}, Shape reconstructed: {reconstructed.shape}")
    loss = criterion(reconstructed, original)
    return loss.mean(dim= (1, 2)).cpu().numpy()


def main(data_path_train, data_path_test_protons, data_path_test_gammas, model_save_dir, epochs=10, batch_size=64, max_events=None, max_events_test=70000):
    # Load dataset

    print(f"Loading dataset from: {data_path_train}")
    start_time = time.time()
    dataset = TelescopeDataset(data_path_train, edge_index, max_events=max_events)
    end_time = time.time()
    print(f"Dataset loaded in {end_time - start_time:.2f} seconds")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, criterion, optimizer
    model = AE().to(device)
    
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    print("Starting training...", flush=True)

    train_loss_history = []
    val_loss_history = []

    for epoch in range(epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index)

            loss_per_pixel = criterion(outputs, batch.x)
            loss = loss_per_pixel.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_dataset)
        train_loss_history.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                outputs = model(batch.x, batch.edge_index)
                loss_per_pixel = criterion(outputs, batch.x)
                loss = loss_per_pixel.mean()
                
                val_loss += loss.item() * batch.num_graphs

        val_loss /= len(val_dataset)
        val_loss_history.append(val_loss)
        epoch_end = time.time()

        print(f"Epoch {epoch+1}/{epochs}, Train Cost: {train_loss:.8f}, Val Cost: {val_loss:.8f}, Duration: {epoch_end - epoch_start:.2f}s", flush=True)

    print("\n=== Computing reconstruction errors on TEST SET ===")

    criterion = nn.MSELoss(reduction='none')

    # Protons test
    dataset_protons = TelescopeDataset(data_path_test_protons, edge_index, max_events=max_events_test)
    original_protons, norm_protons, rec_protons = reconstruct_dataset(model, dataset_protons)
    energies_protons = dataset_protons.get_true_energies()
    err_protons = reconstruction_error(norm_protons, rec_protons, criterion)

    # Gammas test
    dataset_gammas = TelescopeDataset(data_path_test_gammas, edge_index, max_events=max_events_test)
    original_gammas, norm_gammas, rec_gammas = reconstruct_dataset(model, dataset_gammas)
    energies_gammas = dataset_gammas.get_true_energies()
    err_gammas = reconstruction_error(norm_gammas, rec_gammas, criterion)

    print("Median MSE protons:", np.median(err_protons))
    print("Median MSE gammas:", np.median(err_gammas))

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # CSV protons
    df_protons = pd.DataFrame({"error": err_protons})
    df_protons_energy = pd.DataFrame({"true_energy": energies_protons})
    csv_protons_path = os.path.join(model_save_dir, f"errors_protons_{timestamp}.csv")
    csv_protons_energy_path = os.path.join(model_save_dir, f"energies_protons_{timestamp}.csv")
    df_protons.to_csv(csv_protons_path, index=False)
    df_protons_energy.to_csv(csv_protons_energy_path, index=False)
    print(f"Saved protons errors to: {csv_protons_path}")
    print(f"Saved protons energies to: {csv_protons_energy_path}")

    # CSV gammas
    df_gammas = pd.DataFrame({"error": err_gammas})
    df_gammas_energy = pd.DataFrame({"true_energy": energies_gammas})
    csv_gammas_path = os.path.join(model_save_dir, f"errors_gammas_{timestamp}.csv")
    csv_gammas_energy_path = os.path.join(model_save_dir, f"energies_gammas_{timestamp}.csv")
    df_gammas.to_csv(csv_gammas_path, index=False)
    df_gammas_energy.to_csv(csv_gammas_energy_path, index=False)
    print(f"Saved gammas errors to: {csv_gammas_path}")

    model_filename = f"autoencoder_{timestamp}.pth"
    model_save_path = os.path.join(model_save_dir, model_filename)
    
    # Save model
    print(f"Saving model to: {model_save_path}")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_loss_history,
        'val_loss_history': val_loss_history,
        'epochs': epochs,
        'batch_size': batch_size,
        'timestamp': timestamp
    }, model_save_path)
    
    print(f"Model successfully saved as '{model_save_path}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Autoencoder on telescope data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if not os.path.exists(config["model_save_dir"]):
        print(f"Creating directory: {config['model_save_dir']}")
        os.makedirs(config["model_save_dir"], exist_ok=True)

    main(
        data_path_train=config["path_train"],
        data_path_test_protons=config["path_test_protons"],
        data_path_test_gammas=config["path_test_gammas"],
        model_save_dir=config["model_save_dir"],
        epochs=config.get("epochs", 20),
        batch_size=config.get("batch_size", 64),
        max_events=config.get("max_events", None),
        max_events_test=config.get("max_events_test", 70000)
    )
