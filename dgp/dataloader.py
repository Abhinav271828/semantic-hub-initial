import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
from .PCFG import PCFG
from .arith import ArithGrammar
import pickle as pkl


def get_dataloader(
    language: str = "english",  # in ['english', 'expr', 'dyck', 'arith']
    config: dict = {
        "n_nouns": 10,
        "n_verbs": 10,
        "n_adjectives": 10,
        "n_pronouns": 10,
        "n_adverbs": 10,
        "n_conjunctions": 2,
    },  # config for PCFG. see below for other languages.
    replication=(2, 1),  # (D, T)
    alpha: float = 1e5,
    prior_type: str = "dirichlet",
    num_iters: int = 1e6,
    max_sample_length: int = 128,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """Define the PCFG dataloader.

    Args:
        language: The language of the PCFG. One of ['english', 'expr', 'dyck'].
        config: The configuration of the PCFG. The keys depend on the language.
        * For 'english':
            n_nouns: The number of nouns in the vocabulary.
            n_verbs: The number of verbs in the vocabulary.
            n_adjectives: The number of adjectives in the vocabulary.
            n_pronouns: The number of pronouns in the vocabulary.
            n_adverbs: The number of adverbs in the vocabulary.
            n_conjunctions: The number of conjunctions in the vocabulary.
        * For 'expr':
            n_digits: The number of digits in the vocabulary.
            n_ops: The number of operations in the vocabulary.
            bracket: Whether to include brackets in the vocabulary.
        * For 'dyck':
            n_brackets: The number of types brackets in the vocabulary.
        replication (D, T): Specifies replication properties; there are D datatypes with probability softmax([0, -T, ..., -(D-1)T])
        alpha (float, optional): The concentration parameter for the Dirichlet distribution. Defaults to 1e5.
        prior_type (str, optional): The type of prior distribution. Defaults to 'dirichlet'.
        num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
        max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
        seed (int, optional): The random seed. Defaults to 42.
        batch_size (int, optional): The batch size. Defaults to 32.
        num_workers (int, optional): The number of workers. Defaults to 4.

    Returns:
        DataLoader: A pytorch compatible, PCFG dataloader.
    """

    dataset_map = {
        "dyck": PCFGDataset,
        "english": PCFGDataset,
        "expr": PCFGDataset,
        "arith": ArithDataset,
    }

    dataset = dataset_map[language](
        language=language,
        config=config,
        replication=replication,
        alpha=alpha,
        prior_type=prior_type,
        num_iters=num_iters,
        max_sample_length=max_sample_length,
        seed=seed,
    )

    # Create a dataloader
    dataloader = DataLoader(
        dataset,
        sampler=torch.utils.data.RandomSampler(dataset, replacement=True),
        shuffle=False,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return dataloader


class ArithDataset:
    """
    Dataset of basic addition expressions (a + b = c), some in numerals (datatype 0) and some in text (datatype 1).
    """

    def __init__(
        self,
        replication=(2, 1),  # only works with D = 2
        num_iters: int = 1e6,
        max_sample_length: int = 128,
        seed: int = 42,
        **kwargs,
    ):
        """Initialize the ArithDataset.
        Only works with 2 datatypes.

        Args:
            replication (D, T): Specifies replication properties; there are D datatypes with probability softmax([0, -T, ..., -(D-1)T])
            num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
            max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
            seed (int, optional): The random seed. Defaults to 42.
        """
        self.grammar = ArithGrammar(replication, seed)

        # Some setup details
        self.num_iters = int(num_iters)
        self.max_sample_length = max_sample_length
        self.seed = seed

        # Define datatype distribution
        self.datatype_distribution = F.softmax(
            torch.arange(
                start=0,
                end=-replication[0] * replication[1],
                step=-replication[1],
                dtype=torch.float,
            ),
            dim=0,
        )
        self.num_types = replication[0]

        # Build vocabulary
        self.vocab, self.id_to_token_map, self.vocab_size = (
            self.grammar._build_vocabulary()
        )

        # Special tokens
        self.pad_token = "<pad>"
        self.pad_token_id = self.vocab["<pad>"]

    def save_grammar(self, path_to_results: str):
        """
        Save the grammar underlying the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "PCFG.pkl"), "wb") as f:
            pkl.dump(self.grammar, f)

    def load_grammar(self, path_to_results: str):
        """
        Load and override grammar of the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        with open(os.path.join(base_dir, "PCFG.pkl"), "rb") as f:
            self.grammar = pkl.load(f)

    def __len__(self):
        """Return the number of iterations made in the training loop per epoch."""
        return self.num_iters

    def __getitem__(self, index):
        """Get the next sequence from the arithmetic generator.

        Returns:
            sequence: Tensor of token indices
            seq_length: Length of the sequence (excluding padding)
        """
        while True:
            # Sample a datatype
            dtype = torch.multinomial(self.datatype_distribution, 1).squeeze().item()

            # Generate arithmetic sample
            sample = self.grammar.generate_sample(dtype)

            # Tokenize the sequence
            sequence = torch.tensor(self.tokenize_sentence(sample))
            seq_length = float(sequence.size(0))

            # Add BOS and EOS tokens
            sequence = torch.cat(
                (
                    torch.tensor([self.vocab["<bos>"]]),
                    sequence,
                    torch.tensor([self.vocab["<eos>"]]),
                )
            )

            # Check if sequence is too long
            if sequence.size(0) > self.max_sample_length - 10:
                continue  # Regenerate if too long

            # Pad the sequence to max length
            sequence = torch.cat(
                (
                    sequence,
                    torch.tensor(
                        [self.pad_token_id] * (self.max_sample_length - len(sequence))
                    ),
                )
            )
            break

        return sequence, seq_length, dtype


class PCFGDataset:
    def __init__(
        self,
        language: str = "english",  # in ['english', 'expr', 'dyck']
        config: dict = {
            "n_nouns": 10,
            "n_verbs": 10,
            "n_adjectives": 10,
            "n_pronouns": 10,
            "n_adverbs": 10,
            "n_conjunctions": 2,
        },  # config for PCFG. see below for other languages.
        replication=(2, 1),
        alpha: float = 1e5,
        prior_type: str = "dirichlet",
        num_iters: int = 1e6,
        max_sample_length: int = 128,
        seed: int = 42,
        **kwargs,
    ):
        """Define the PCFG dataset.

        Args:
            language: The language of the PCFG. One of ['english', 'expr', 'dyck1', 'dyck2'].
            config: The configuration of the PCFG. The keys depend on the language.
            * For 'english':
                n_nouns: The number of nouns in the vocabulary.
                n_verbs: The number of verbs in the vocabulary.
                n_adjectives: The number of adjectives in the vocabulary.
                n_pronouns: The number of pronouns in the vocabulary.
                n_adverbs: The number of adverbs in the vocabulary.
                n_conjunctions: The number of conjunctions in the vocabulary.
            * For 'expr':
                n_digits: The number of digits in the vocabulary.
                n_ops: The number of operations in the vocabulary.
                bracket: Whether to include brackets in the vocabulary.
            * For 'dyck':
                n_brackets: The number of types brackets in the vocabulary.
            replication (D, T): Specifies replication properties; there are D datatypes with probability softmax([0, -T, ..., -(D-1)T])
            alpha (float, optional): The concentration parameter for the Dirichlet distribution. Defaults to 1e5.
            prior_type (str, optional): The type of prior distribution. Defaults to 'dirichlet'.
            num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
            max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
            seed (int, optional): The random seed. Defaults to 42.

        Returns:
            PCFGDataset: A PCFG dataset.
        """

        # Some setup details
        self.num_iters = int(num_iters)
        self.max_sample_length = max_sample_length

        # Define the PCFG
        self.PCFG = PCFG(
            language=language,
            config=config,
            num_types=replication[0],
            alpha=alpha,
            prior_type=prior_type,
            seed=seed,
        )
        self.vocab_size = self.PCFG.vocab_size

        self.datatype_distribution = F.softmax(
            torch.arange(
                start=0,
                end=-replication[0] * replication[1],
                step=-replication[1],
                dtype=torch.float,
            ),
            dim=0,
        )

        ## Special tokens
        # Pad token
        self.pad_token = "<pad>"
        self.pad_token_id = self.PCFG.vocab["<pad>"]

        # Define the PCFG generator
        self.generator = self.PCFG.sentence_generator(num_of_samples=self.num_iters)

    def save_grammar(self, path_to_results: str):
        """
        Save the grammar underlying the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        os.makedirs(base_dir, exist_ok=True)
        with open(os.path.join(base_dir, "PCFG.pkl"), "wb") as f:
            pkl.dump(self.PCFG, f)

    def load_grammar(self, path_to_results: str):
        """
        Load and override grammar of the dataset
        """
        base_dir = os.path.join(path_to_results, "grammar")
        with open(os.path.join(base_dir, "PCFG.pkl"), "rb") as f:
            self.PCFG = pkl.load(f)

    def __len__(self):
        """
        Return the number of iterations made in the training loop per epoch.
        """
        return self.num_iters

    def __getitem__(self, index):
        """
        Get the next sequence from the PCFG generator.
        """

        while True:
            try:
                sequence = self.generator.__next__()
            except StopIteration:
                # regenerate for the next epoch’s worth of samples
                self.generator = self.PCFG.sentence_generator(
                    num_of_samples=self.num_iters
                )
                continue

            dtype = torch.multinomial(self.datatype_distribution, 1).squeeze().item()

            # Tokenize the sequence
            sequence = torch.tensor(self.PCFG.tokenize_sentence(sequence, dtype))
            seq_length = float(sequence.size(0))

            sequence = torch.cat(
                (
                    torch.tensor([self.PCFG.vocab["<bos>"]]),
                    sequence,
                    torch.tensor([self.PCFG.vocab["<eos>"]]),
                )
            )

            # Truncate the sequence if it is longer than the max sequence length
            if sequence.size(0) > self.max_sample_length - 10:
                pass

            # Pad the sequence to the max sequence length with <pad>
            else:
                sequence = torch.cat(
                    (
                        sequence,
                        torch.tensor(
                            [self.pad_token_id]
                            * (self.max_sample_length - len(sequence))
                        ),
                    )
                )
                break

        return sequence, seq_length, dtype
