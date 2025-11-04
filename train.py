import hydra
from nltk import data
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import GPT
from dgp import get_dataloader
from evals import grammar_evals, arith_evals

from utils import init_wandb, set_seed, save_config, open_log, cleanup
from utils import sanity_checks, configure_optimizers, update_cosine_warmup_lr
from utils import save_model, move_to_device, log_train, log_eval


@hydra.main(config_path="./config", config_name="conf.yaml", version_base="1.3")
def main(cfg):
    init_wandb(
        cfg,
        project_name="sem-hub",
    )
    set_seed(cfg.seed)
    save_config(cfg)
    open_log(cfg)
    device = cfg.device  # if torch.cuda.is_available() else "cpu"

    # Dataloader
    dataloader = get_dataloader(
        language=cfg.data.language,
        config=cfg.data.config,
        replication=(cfg.data.D, cfg.data.T),
        alpha=cfg.data.alpha,
        prior_type=cfg.data.prior_type,
        num_iters=cfg.data.num_iters * cfg.data.batch_size,
        max_sample_length=cfg.data.max_sample_length,
        seed=cfg.seed,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Check if model is compatible with data
    sanity_checks(cfg, dataloader.dataset.max_sample_length)

    # Define model
    model = GPT(cfg.model, dataloader.dataset.vocab_size)
    model.to(device)
    if cfg.model.compile:
        model = torch.compile(model)
    print("number of parameters: %.2fM" % (model.get_num_params() / 1e6,))
    print("device:", next(model.parameters()).device)

    # Optimizer
    optimizer = configure_optimizers(model, cfg.optimizer)

    # Train
    train(cfg, model, dataloader, optimizer, device)

    # Close wandb and log file
    cleanup(cfg)


def train(cfg, model, dataloader, optimizer, device):
    """
    Training function
    """
    # Set model to train mode
    model.train()

    # Set to True to save the grammar underlying the dataset
    save_grammar = True

    # Data type (bf16 for efficiency)
    dt = torch.bfloat16 if cfg.bf16 else torch.float32

    # Configuration
    total_steps = len(dataloader) * cfg.epochs

    loss, lr, it, save_tables = 0.0, 0.0, 0, 0
    print("Total training steps: ", total_steps)
    print("Learning rate warmup steps: ", cfg.data.num_iters)

    # Save initial model
    results_dir = save_model(cfg, model, optimizer, it)

    # Load pretrained model if desired
    if cfg.model.use_pretrained:
        model.load_state_dict(torch.load(cfg.model.pretrain_dir)["net"])
        optimizer.load_state_dict(torch.load(cfg.model.pretrain_dir)["optimizer"])

    # Training loop
    for e in range(cfg.epochs):
        for sequences, seq_lengths, dtypes in tqdm(dataloader, desc=f"Epoch {e}"):
            # Split sequences into inputs and labels
            # (B, L) -> (B, L-1), (B, L-1)
            B = sequences.size(0)
            inputs, labels = move_to_device(
                [sequences[:, :-1], sequences[:, 1:]], device
            )

            # Sequence statistics to log as well
            train_lengths = {
                "max": seq_lengths.max().item(),
                "min": seq_lengths.min().item(),
                "mean": seq_lengths.mean().item(),
            }

            # Log train metrics
            if it % cfg.log.log_interval == 0:
                log_train(it, cfg.deploy, lr, loss, train_lengths)

            # Evals
            if it % cfg.log.eval_interval == 0:
                model.eval()  # Set to eval mode

                if cfg.data.language in ["expr", "dyck", "english"]:
                    # Grammaticality results
                    grammar_results_dict = (
                        grammar_evals(
                            cfg=cfg,
                            model=model,
                            grammar=dataloader.dataset.PCFG,
                            device=device,
                            print_samples=cfg.log.print_gen_samples,
                        )
                        if cfg.eval.grammar
                        else (None, None)
                    )
                else:
                    # Arith validity results
                    grammar_results_dict = (
                        arith_evals(
                            cfg=cfg,
                            model=model,
                            dataset=dataloader.dataset,
                            device=device,
                            print_samples=cfg.log.print_gen_samples,
                        )
                        if cfg.eval.grammar
                        else (None, None)
                    )

                # Log eval metrics
                save_tables = log_eval(
                    deploy=cfg.deploy,
                    it=it,
                    save_tables=save_tables,
                    grammaticality_results=grammar_results_dict,
                )

                model.train()  # Set back to train mode

            # Update LR
            it, lr = update_cosine_warmup_lr(it, cfg.optimizer, optimizer, total_steps)

            # Compute loss
            optimizer.zero_grad(set_to_none=True)  # Set gradients to None
            with torch.amp.autocast(
                device_type=("cuda" if "cuda" in device else "cpu"),
                dtype=dt,
            ):  # Mixed precision
                # Print a few human-readable input sequences before forward pass

                logits = model(inputs)  # (B, L-1, V)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    # CORR: To allow us to avoid the replacement above.
                    ignore_index=dataloader.dataset.pad_token_id,  # -100,
                    reduction="none",
                )  # (B*L-1)
                loss = loss.reshape(B, -1).sum(dim=1)  # Sum over sequence length

                loss = loss.mean()  # Average loss

            # Update model
            loss.backward()  # Compute gradients
            if cfg.optimizer.grad_clip > 0.0:  # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.optimizer.grad_clip
                )
            optimizer.step()  # Update weights

            # Save model every few iterations
            if it % cfg.log.save_interval == 0:
                save_model(cfg, model, optimizer, it)
            if save_grammar:
                dataloader.dataset.save_grammar(results_dir)
                save_grammar = False  # Save only once

        # Save after every epoch
        save_model(cfg, model, optimizer, it)


if __name__ == "__main__":
    main()
