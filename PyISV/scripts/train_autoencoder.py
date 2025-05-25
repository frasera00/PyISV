# This script trains an autoencoder model using PyTorch and the PyISV library.

import os, datetime, time, logging, warnings, numpy as np, shutil
warnings.filterwarnings("ignore")

import torch, torch.distributed as dist
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Optional

from PyISV.utils.IO_utils import (
    load_tensor, setup_logging, is_main_process
)
from PyISV.utils.training_utils import (
    get_device, get_data_loader, apply_cuda_optimizers,
    setup_lr_scheduler_with_warmup, train_epoch_step,
    EarlyStopping, SaveBestModel, Dataset, PreloadedDataset
)
from PyISV.utils.validation_utils import Validator   
from PyISV.utils.set_architecture import import_config
from PyISV.utils.define_root import PROJECT_ROOT as root_dir
from PyISV.neural_network import NeuralNetwork 

# Profiling utilities for performance analysis
import torch.utils.bottleneck, torch.profiler

class Trainer():
    def __init__(self, 
        params_dict: dict | None = None, 
        json_file: str | None = None,   
        run_id: Optional[str | int] = None,
        models_dir: Optional[str] = None,
        logging: Optional[bool] = True,
        debug: Optional[bool] = False) -> None:

        # Import config
        if params_dict is not None:
            self.config = import_config(param_dict=params_dict)
        elif json_file is not None:
            self.config = import_config(json_file=json_file)
        else:
            raise ValueError("Either param_dict or json_file must be provided.")
        
        # Set utility parameters
        self.run_id = run_id if run_id is not None else datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.debug = debug
        self.logs = logging

        # Extract params from config and optimize for CUDA (if available)
        self._extract_params()
        apply_cuda_optimizers(device=self.device)

        # Configure paths, dirs and files
        self.root_dir = root_dir
        self.models_dir = f"{models_dir}/{self.run_id}" if models_dir is not None else f"{self.root_dir}/models/{self.run_id}"
        self._define_paths()
        self._setup_dirs()
        self._setup_files()

        if self.logs:
            setup_logging(f"{self.logs_dir}/train.log")

        self.validator = Validator(self.config)

        if self.use_ddp:
            self._setup_ddp()
        
        if is_main_process():
            print(f"\nTraining on {self.device.type} with run ID: {self.run_id}\n")

    def prepare_data(self, verbose: bool=True) -> None:
        input_file = self.config['INPUTS']['dataset']
        target_file = self.config['INPUTS']['target']
        input_data = load_tensor(input_file)
        target_data = load_tensor(target_file) if target_file else input_data.clone()
        
        # DEBUG: Verify alignment
        if self.debug:
            print(f"Input shape: {input_data.shape}")
            print(f"Target shape: {target_data.shape}")
            
            # Check if they're accidentally the same
            if torch.equal(input_data, target_data):
                print("⚠️ WARNING: Input and target are identical!")
            
            # Check first sample alignment
            print(f"Sample 0 input mean: {input_data[0].mean():.6f}")
            print(f"Sample 0 target mean: {target_data[0].mean():.6f}")

        dataset = Dataset(
            input_data, target_data,
            norm_inputs=True, norm_targets=True,
            norm_mode=self.norm, device=self.device
        )

        # DEBUG: Check normalized data
        if self.debug:
            print(f"Normalized input stats: mean={dataset.inputs.mean():.6f}, std={dataset.inputs.std():.6f}")
            print(f"Normalized target stats: mean={dataset.targets.mean():.6f}, std={dataset.targets.std():.6f}")
        
        # Save normalization params
        for key, value in self.norm_files.items():
            if hasattr(dataset, key):
                ds = getattr(dataset, key)
                np.save(value, ds.detach().cpu().numpy()) # type: ignore

        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_valid, y_train, y_valid = train_test_split(
            dataset.inputs, dataset.targets,
            train_size=self.train_sz,
            random_state=self.seed,
            shuffle=True, stratify=None
        )

        # Loads the data using pre-computed normalization
        self.train_dataset = PreloadedDataset(X_train, y_train)
        self.valid_dataset = PreloadedDataset(X_valid, y_valid)

        # Save the validation dataset
        #torch.save(self.valid_dataset.inputs.detach().cpu(), 
        #           f"{self.outputs_dir}/input_validation_data.pt")

        self.train_loader = get_data_loader(
            self.train_dataset,
            batch_size=self.batch_sz,
            num_workers=self.n_wks,
            pin_memory=self.pin_mem,
            use_ddp=self.use_ddp,
            shuffle=True,
        )
        self.valid_loader = get_data_loader(
            self.valid_dataset,
            batch_size=self.batch_sz,
            num_workers=self.n_wks,
            pin_memory=self.pin_mem,
            use_ddp=self.use_ddp,
            shuffle=False,
        )

    def prepare_model(self, evaluation_mode: bool = False, verbose: bool = True) -> None:
        # Instantiate model
        self.model_dict = self.config['MODEL']
        self.input_shape = self.config['MODEL']['input_shape']
        self.model = NeuralNetwork(self.model_dict).to(self.device)

        # Use PyTorch optimizations for CPU
        if self.device.type == "cpu":
            try:
                import intel_extension_for_pytorch as ipex # type: ignore
                self.model = ipex.optimize(self.model)
                if is_main_process():
                    print("ℹ️  Using IPEX optimizations for CPU model")
            except ImportError:
                if is_main_process():
                    print("ℹ️  IPEX not available, using standard CPU optimizations")
                # Apply other CPU optimizations
                torch._C._jit_set_profiling_executor(True)
                torch._C._jit_set_profiling_mode(True)
            
        # DDP/DataParallel
        if self.use_ddp and dist.is_initialized():
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
            torch.cuda.set_device(local_rank)
            if self.device.type == "cuda":
                self.model = DDP(self.model, device_ids=[local_rank])
            else:
                self.model = DDP(self.model)
        elif (torch.cuda.is_available() and torch.cuda.device_count() > 1 and 
            str(self.device).startswith("cuda") and self.use_paral):
            self.model = DataParallel(self.model)
            if is_main_process():
                print(f"ℹ️  Using DataParallel with {torch.cuda.device_count()} GPUs")
        else:
            print(f"ℹ️  Using single GPU: {self.device}")
        
        # Loss and optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Callbacks
        if not evaluation_mode:
            if self.has_early_stop:
                early_stop_params = self.config['TRAINING']['early_stopping_params']
                self.early_stop = EarlyStopping(**early_stop_params)
            self.save_best_model = SaveBestModel(best_model_name=self.model_file)

    def save_best_model_if_better_loss(self, epoch: int, val_loss: float, total_loss: float) -> None:
        if (self.best_loss is None) or (val_loss < self.best_loss):
            self.save_best_model(model=self.model, current_valid_loss=val_loss,
                current_train_loss=total_loss,epoch=epoch, optimizer=self.optimizer)
        self.best_loss = val_loss

    def step_lr(self, lr_scheduler: "torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.LRScheduler") -> float:
        current_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()
        return current_lr
    
    def train(self, start_epoch: int = 0, verbose: bool = True, log_interval: int = 10) -> None:
        if verbose and is_main_process():
            print(f"ℹ️ Model type: {type(self.model)}")
            print(f"ℹ️ Device: {self.device}")
            print(f"ℹ️ Use DDP: {self.use_ddp}")
            print(f"ℹ️ Train loader length: {len(self.train_loader)}")
            print(f"ℹ️ Loss function: {self.loss_fn}")

        self.best_loss = None

        if self.sch_lr:
            self.sch_params = self.config['TRAINING']['scheduled_params']
            lr_scheduler = setup_lr_scheduler_with_warmup(
                optimizer=self.optimizer,
                params=self.sch_params
            )

        scaler = torch.amp.GradScaler(enabled=True) # type: ignore

        if verbose and is_main_process():
            print(f"\n▶️ Starting training from epoch {start_epoch} to {self.max_epcs}\n")

        learn_rate = []
        stats = {}
        t0 = time.time()
        for epoch in range(start_epoch, self.max_epcs):
            if self.sch_lr:
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch+1}] Learning rate: {current_lr:.6e}")

            total_loss = train_epoch_step(
                model=self.model,
                optimizer=self.optimizer,
                scaler=scaler,
                epoch=epoch,
                device=self.device,
                data_loader=self.train_loader,
                loss_function=self.loss_fn,
                use_ddp=self.use_ddp,
            )

            val_loss = self.validator.validate_epoch(
                model=self.model, device=self.device,
                data_loader=self.valid_loader,
                loss_function=self.loss_fn,
            )

            if self.has_early_stop:
                should_stop = False
                if epoch >= self.min_epcs:
                    if is_main_process():
                        should_stop = self.early_stop(val_loss)
                    if self.use_ddp and torch.distributed.is_initialized():
                        stop_tensor = torch.tensor(int(should_stop), device=self.device)
                        torch.distributed.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())
                    if should_stop:
                        if verbose and is_main_process():
                            print(f'Early stopping at epoch {epoch + 1}')
                        if logging and is_main_process():
                            logging.log(logging.INFO, f'Early stopping at epoch {epoch + 1}')
                        break

            if self.stats_file:
                stats['epoch'] = epoch + 1
                stats['train_loss'] = total_loss
                stats['val_loss'] = val_loss
                stats['learning_rate'] = current_lr if self.sch_lr else self.optimizer.param_groups[0]['lr']
                stats['time_per_epoch'] = time.time() - t0

            if self.sch_lr:
                current_lr = self.step_lr(lr_scheduler)
                learn_rate.append(current_lr)

            if ((epoch + 1) % log_interval == 0):
                elapsed_time = time.time() - t0
                t0 = time.time()

                if logging and is_main_process():
                    logging.log(logging.INFO, f'Epoch {epoch + 1} - validation loss: {val_loss:.4f}')
                
                if is_main_process():
                    self.save_best_model_if_better_loss(epoch, val_loss, total_loss)
                time_per_epoch = elapsed_time / log_interval

                print(f'⏳ [Epoch {epoch + 1}] - train loss: {total_loss:.4f} - validation loss: {val_loss:.4f} - ({time_per_epoch:.2f}s/epoch)')
        if verbose and is_main_process():
            print(f"\n Training completed. Best validation loss: {self.best_loss:.4f}")
        
        if self.stats_file and is_main_process():
            # Save training stats
            stats['best_loss'] = self.best_loss
            stats['learn_rate'] = learn_rate
            np.savez(self.stats_file, **stats)
            if verbose:
                print(f"Training statistics saved to {self.stats_file}")

    def run_training(self, verbose: bool=True) -> None:
        self.prepare_data(verbose=verbose)
        self.prepare_model(verbose=verbose)
        self.train(verbose=verbose)
        self._cleanup()

    def _extract_params(self) -> None:
        try:
            # Set general parameters
            self.device = get_device(device=self.config['GENERAL']['device'])
            self.use_ddp = self.config['GENERAL']['use_ddp']
            self.use_paral = self.config['GENERAL']['use_data_parallel']
            self.seed = self.config['GENERAL']['seed']

            # Set training parameters
            self.norm = self.config['TRAINING']['normalization']
            self.train_sz = self.config['TRAINING']['train_size']
            self.batch_sz = self.config['TRAINING']['batch_size']
            self.n_wks = self.config['TRAINING']['num_workers']
            self.pin_mem = self.config['TRAINING']['pin_memory']
            self.loss_fn = self.config['TRAINING']['loss_function']
            self.min_epcs = self.config['TRAINING']['min_epochs']
            self.max_epcs = self.config['TRAINING']['max_epochs']
            self.lr = self.config['TRAINING']['learning_rate']
            self.sch_lr = self.config['TRAINING']['scheduled_lr']
            self.has_early_stop = self.config['TRAINING']['early_stopping']
        except KeyError as e:
            raise KeyError(f"Missing parameter in config: {e}")

    def _setup_files(self) -> None:
        # Set file names
        self.model_file = f"{self.models_dir}/model.pt"
        self.stats_file = f"{self.stats_dir}/stats.dat"
        self.arch_file = f"{self.models_dir}/arch.dat"
        self.norm_files = {
            "subval_inputs": f"{self.norms_dir}/subval_inputs.npy",
            "divval_inputs": f"{self.norms_dir}/divval_inputs.npy",
            "subval_targets": f"{self.norms_dir}/subval_targets.npy",
            "divval_targets": f"{self.norms_dir}/divval_targets.npy"
        }

    def _setup_dirs(self) -> None:
        # Create directories if they don't exist
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.norms_dir, exist_ok=True)
        os.makedirs(self.stats_dir, exist_ok=True)

    def _define_paths(self) -> None:
        # Define paths for model, logs, and outputs
        self.logs_dir = f"{self.models_dir}/logs/"
        self.outputs_dir = f"{self.models_dir}/outputs"
        self.norms_dir = f"{self.models_dir}/norms"
        self.stats_dir = f"{self.models_dir}/stats"

    def _cleanup(self) -> None:
        if self.use_ddp:
            dist.destroy_process_group()
            torch.cuda.empty_cache()
    
    def _setup_ddp(self) -> None:
        """Setup Distributed Data Parallel (DDP) configuration"""
        
        if not self.use_ddp:
            self._setup_single_process()
            return
        
        # Get DDP environment variables
        ddp_env = self._get_ddp_environment()
        local_rank, global_rank, world_size = ddp_env
        
        if self.debug:
            print(f"[RANK {global_rank}] LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}, "
                  f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
        
        # Setup device and backend
        backend = self._setup_device_and_backend(local_rank)
        
        # Initialize process group
        try:
            dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=30))
            torch.distributed.barrier()
            if is_main_process():
                print(f"✅ DDP initialized successfully with {world_size} processes using {backend} backend")
        except Exception as e:
            print(f"❌ Failed to initialize DDP: {e}")
            raise

    def _get_ddp_environment(self) -> tuple[int, int, int]:
        """Extract DDP environment variables from torchrun or SLURM"""
        # Torchrun sets these automatically
        if 'LOCAL_RANK' in os.environ:
            local_rank = int(os.environ['LOCAL_RANK'])
            global_rank = int(os.environ.get('RANK', 0))
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            print(f"[DDP] Detected torchrun: LOCAL_RANK={local_rank}, RANK={global_rank}, WORLD_SIZE={world_size}")
            return local_rank, global_rank, world_size
        
        # SLURM fallback
        if 'SLURM_PROCID' in os.environ:
            local_rank = int(os.environ.get('SLURM_LOCALID', 0))
            global_rank = int(os.environ.get('SLURM_PROCID', 0))
            world_size = int(os.environ.get('SLURM_NTASKS', 1))
            
            # Set environment variables for consistency
            os.environ.update({
                'RANK': str(global_rank),
                'LOCAL_RANK': str(local_rank),
                'WORLD_SIZE': str(world_size)
            })
            
            self._setup_slurm_networking()
            return local_rank, global_rank, world_size
        
        # Single process fallback (shouldn't happen with proper DDP setup)
        print("⚠️ WARNING: No DDP environment detected, falling back to single process")
        return 0, 0, 1

    def _setup_slurm_networking(self) -> None:
        """Setup SLURM-specific networking configuration"""
        
        if 'MASTER_ADDR' not in os.environ:
            try:
                import subprocess
                result = subprocess.run(
                    ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    os.environ['MASTER_ADDR'] = result.stdout.strip().split('\n')[0]
                else:
                    print(f"⚠️ Failed to get SLURM hostnames: {result.stderr}")
                    os.environ['MASTER_ADDR'] = 'localhost'
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                print(f"⚠️ SLURM command failed: {e}, using localhost")
                os.environ['MASTER_ADDR'] = 'localhost'
        
        if 'MASTER_PORT' not in os.environ:
            # Generate deterministic port based on job ID
            job_id = int(os.environ.get('SLURM_JOBID', '0'))
            port = 29500 + (job_id % 1000)  # Port range 29500-30499
            os.environ['MASTER_PORT'] = str(port)

    def _setup_device_and_backend(self, local_rank: int) -> str:
        """Setup compute device and communication backend"""
        
        if torch.cuda.is_available():
            # GPU setup
            if local_rank >= torch.cuda.device_count():
                raise RuntimeError(f"Local rank {local_rank} >= available GPUs {torch.cuda.device_count()}")
            
            # Only set device if not already set
            current_device = torch.cuda.current_device()
            if current_device != local_rank:
                torch.cuda.set_device(local_rank)
            self.device = torch.device(f'cuda:{local_rank}')
            backend = self._get_gpu_backend()
            
            #print(f"GPU Memory, rank {local_rank}: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.1f} GB")
            if is_main_process():
                print(f"ℹ️ Using GPU backend: {backend}")
        else:
            # CPU setup
            self.device = torch.device('cpu')
            backend = "gloo"  # Only gloo supports CPU
            
            # CPU thread optimization
            num_threads = int(os.environ.get('OMP_NUM_THREADS', os.cpu_count() or 1))
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(min(4, num_threads))  # Reasonable default
            
            if is_main_process():
                print(f"Using CPU with {num_threads} threads and {backend} backend")
        
        return backend

    def _get_gpu_backend(self) -> str:
        """Determine the best GPU communication backend"""
        # Try Intel CCL first (for Intel GPUs)
        try:
            import intel_extension_for_pytorch # type: ignore
            import oneccl_bindings_for_pytorch # type: ignore
            return "ccl"
        except ImportError:
            pass
        
        # Check for NCCL availability
        if hasattr(torch.distributed, 'is_nccl_available') and torch.distributed.is_nccl_available():
            return "nccl"
        
        # Fallback to gloo
        if is_main_process():
            print("⚠️ NCCL not available, falling back to gloo (may be slower)")
        return "gloo"

    def _setup_single_process(self) -> None:
        """Setup for single process training (no DDP)"""
        if torch.cuda.is_available():
            # Use first available GPU
            self.device = torch.device('cuda:0')
            if is_main_process():
                print(f"Single process training on {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            num_threads = os.cpu_count() or 1
            torch.set_num_threads(num_threads)
            torch.set_num_interop_threads(min(4, num_threads))
            if is_main_process():
                print(f"Single process training on CPU with {num_threads} threads")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train an autoencoder model using PyTorch.')
    parser.add_argument(
        '-c','--config',
        dest='json_file',
        type=str,
        required=True,
        help='Path to the JSON configuration file'
    )
    parser.add_argument(
        '-m','--models_dir',
        dest='models_dir',
        type=str,
        help='Directory where models/checkpoints will be saved'
    )
    parser.add_argument(
        '-r','--run_id',
        dest='run_id',
        type=str,
        help='Optional run identifier (defaults to timestamp)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        dest='debug',
        help='Turn on debug printing'
    )

    args = parser.parse_args()

    trainer = Trainer(
        params_dict=None,
        json_file=args.json_file,
        run_id=args.run_id,
        models_dir=args.models_dir,
        debug=args.debug
    )

    trainer.run_training()