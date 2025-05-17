# This script evaluates the autoencoder model by plotting the training loss curve,
# reconstructing the input data, and visualizing the latent space using t-SNE.

import datetime, argparse, warnings, json, torch
import numpy as np, pandas as pd, matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from typing import Optional 
from pathlib import Path
from tqdm import tqdm

from PyISV.utils.set_architecture import import_config
from PyISV.utils.training_utils import get_device, load_tensor
from scripts.train_autoencoder import Trainer
from PyISV.utils.define_root import PROJECT_ROOT as root_dir

class Evaluator(Trainer):
    """Class to evaluate the autoencoder model."""
    def __init__(self, config: str | Path,
                 run_id: str, 
                 models_dir: Optional[str] = None,
                 device: Optional[str] = None) -> None:
        
        # Import config
        self.run_id = run_id
        self.config = import_config(config)
        self.input_shape = self.config["MODEL"]["input_shape"]
        self.device = device if device else get_device(self.config["GENERAL"]["device"])
        
        # Set paths to directories
        self.root_dir = root_dir
        self.models_dir = models_dir if models_dir else f"{self.root_dir}/models/{self.run_id}"
        self._define_paths()
        self._define_files()

        self.eval_folder = f"{self.models_dir}/evaluation"
        Path(self.eval_folder).mkdir(exist_ok=True)

        Trainer.prepare_model(self, evaluation_mode=True)
        self.model = self.model.to(self.device)

    def _define_files(self) -> None:
        self.arch_file = f"{self.models_dir}/arch.dat"
        self.model_file = f"{self.models_dir}/model.pt"
        self.stats_file = f"{self.stats_dir}/stats.dat"
        self.inputs_file = f"{self.inputs_dir}/{self.config['INPUTS']['dataset']}"

    def _define_paths(self) -> None:
        self.data_dir = f"{self.root_dir}/datasets"
        self.inputs_dir = f"{self.data_dir}/RDFs"
        self.norms_dir = f"{self.models_dir}/norms"
        self.stats_dir = f"{self.models_dir}/stats"
        self.outputs_dir = f"{self.models_dir}/outputs"

    def plot_loss_curve(self) -> None:
        """Plot the training and validation loss curves and optionally save to output_path."""
        df = pd.read_csv(self.stats_file)
        fig, ax = plt.subplots(1,1)
        ax.plot(df['epoch'], df['train_loss'], label='train')
        ax.plot(df['epoch'], df['val_loss'],   label='val')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss'); ax.legend()
        if self.outputs_dir:
            fig.savefig(f"{self.outputs_dir}/loss_curve.png")
        return

    def export_onnx(self) -> None:
        """Export the model to ONNX format."""
        import torch.onnx
        dummy_input = torch.randn(1, *self.input_shape, device=self.device)
        torch.onnx.export(
            self.model, # type: ignore
            dummy_input, # type: ignore
            f"{self.models_dir}/model.onnx",
            input_names=['input'],
            output_names=['output'],
            opset_version=11
        )
        print(f'Exported ONNX model to: {self.models_dir}/model.onnx')
        return
    
    def plot_loss_errors(self, errors: np.ndarray) -> None:
        # Histogram of errors and t-SNE
        fig, ax = plt.subplots()
        ax.hist(errors, bins=50, alpha=0.7)
        ax.set_xlabel('Reconstruction RMSE'); ax.set_ylabel('Count')
        fig.savefig(f"{self.eval_folder}/reconstruction_errors.png")

    def plot_TSNE(self, embeds: np.ndarray) -> np.ndarray:
        from openTSNE import TSNE

        tsne = TSNE(n_jobs=4, random_state=0)
        z2d = np.array(tsne.fit(embeds))
        fig, ax = plt.subplots()
        ax.scatter(z2d[:,0], z2d[:,1], s=5, alpha=0.6)
        ax.set_title('Latent space t-SNE'); plt.tight_layout()
        fig.savefig(f"{self.eval_folder}/latent_tsne.png")
        return z2d

    def evaluate_model(self) -> None:
        """Evaluate the model's reconstructions, plot latent space, and save results to output_dir."""

        onnx_path = f"{self.eval_folder}/model.onnx"
        self.export_onnx()  # Export the model to ONNX format
        self.model.eval()

        # Load saved inputs
        inputs = load_tensor(self.inputs_file)
        print(f"\nLoaded inputs with shape: {inputs.shape}\n")
        
        # Process the data in batches to avoid memory issues
        batch_size = 128
        all_errors, embeddings, outputs = [], [], []
        
        with torch.no_grad():
            for i in tqdm(range(0, inputs.shape[0], batch_size), desc="Evaluating batches"):                    
                # Forward pass
                batch = inputs[i:i+batch_size].to(self.device)
                recon, latent = self.model(batch)
                
                # Calculate per-sample MSE errors (no reduction)
                mse_per_sample = torch.nn.functional.mse_loss(recon, batch, reduction='none')
                errs = mse_per_sample.mean(dim=(1, 2)).detach().cpu().numpy()
                flat_latent = latent.detach().cpu().reshape(latent.shape[0], -1)
                
                all_errors.append(errs)
                embeddings.append(flat_latent.numpy())
                outputs.append(recon.detach().cpu().numpy())
                
                # Print stats for first batch only
                if i == 0:
                    print(f"Latent shape: {latent.shape}")
                    print(f"Latent min/max/mean: {latent.min().item():.5f}, {latent.max().item():.5f}, {latent.mean().item():.5f}")
                    print(f"Flat latent shape: {flat_latent.shape}")
        
        # Concatenate results
        errors = np.concatenate(all_errors)
        embeds = np.concatenate(embeddings, axis=0)
        outputs = np.concatenate(outputs, axis=0)

        # Unnormalize outputs
        outsubval = np.load(self.norm_files['subval'])
        outdivval = np.load(self.norm_files['divval'])
        outputs_denorm = torch.from_numpy( (outputs * outdivval) + outsubval )

        self.plot_loss_errors(errors)
        z2d = self.plot_TSNE(embeds)

        # Save results
        torch.save(outputs_denorm, f"{self.eval_folder}/reconstructed_outputs.pt")
        np.save(f"{self.eval_folder}/tsne.npy", z2d)
        np.save(f"{self.eval_folder}/reconstructed_errors.npy", errors)
        np.save(f"{self.eval_folder}/embeddings.npy", embeds)

        # Save evaluation summary as JSON
        self.save_evaluation_summary(errors, embeds)
        print(f"Evaluation results saved to: {self.eval_folder}")
        return

    def save_evaluation_summary(self, errors: np.ndarray, embeddings: np.ndarray) -> None:
        # Create summary dictionary with statistics
        eval_summary = {
            "run_id": self.run_id,
            "evaluation_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error_stats": {
                "mean": float(np.mean(errors)),
                "median": float(np.median(errors)),
                "std": float(np.std(errors)),
                "min": float(np.min(errors)),
                "max": float(np.max(errors)),
                "count": int(len(errors))
            },
            "latent_space": {
                "dimensions": int(embeddings.shape[1]),
                "samples": int(embeddings.shape[0])
            },
            "model_info": {
                "config_path": str(Path(self.arch_file).parent),
                "architecture": str(Path(self.arch_file).name)
            }
        }
        
        # Save to JSON file
        output_file = f"{self.outputs_dir}/eval_summary.json"
        with open(output_file, 'w') as f:
            json.dump(eval_summary, f, indent=4)
        
        print(f"\nEvaluation summary saved to: {output_file}")
        return

# -------------- Execute the main function --------------- #

if __name__ == "__main__":
    # Parse run ID from command line
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', '-r',type=str)
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--models_dir', '-m', type=str)
    args = parser.parse_args()


    evaluator = Evaluator(config=args.config, run_id=args.run_id, models_dir=args.models_dir)
    evaluator.plot_loss_curve()
    evaluator.evaluate_model()

    print("\nEvaluation completed successfully.")
