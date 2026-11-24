#!/usr/bin/env python3
# pylint: disable=line-too-long

"""
Feed-forward network to predict a player's pre-flop action (fold, check/call, raise).
Just an evaluation, not feature dense.

Reads pre-flop rows from the decisions/ table written by phh_to_parquet.py. Each row becomes 12 numbers:
    cards | known flag, high rank, low rank, suited, pair (all 0 when cards are hidden)
    table | seat (0 = small blind), n_players, n_active, n_raises
    chips | pot, to_call, and stack

Basic usage:
    ./scripts/feed_forward_preflop.py
"""

# std libs
import sys
import random
from time import time
from pathlib import Path
from functools import wraps
from argparse import ArgumentParser
from dataclasses import dataclass, field

# external libs
import torch
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

##### global constants ####################
COLUMNS = ["hand_id", "seat", "n_players", "n_active", "big_blind", "pot", "to_call", "stack", "n_raises", "hole1", "hole2", "action"]
ACTION_NAMES = ["fold", "call", "raise"]

# to add others, put high card to the left, not resolving high card like in make_features
DEFAULT_SAMPLE_HANDS = {
    "AA": (51, 50),
    "AKs": (51, 47),
    "JTs": (39, 35),
    "72o": (20, 1),
    "??": (-1, -1)
}

##### funcs ####################

def timer(func):
    """ Generic decorator for funcs that use return (not yield) """

    @wraps(func) # required to preserve original class metadata - https://stackoverflow.com/questions/308999/what-does-functools-wraps-do
    def executor(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        print(f"`{func.__name__}()` took {time() - start_time:.2f}s")
        return result
    return executor


def get_args():
    """ Argparser retrieval """

    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--data_path", type = Path, help = "path to the decisions parquet table", default = Path("parquet/decisions"))
    parser.add_argument("--max_rows", type = int, help = "total pre-flop rows to read", default = None)
    parser.add_argument("--section_rows", type = int, help = "max pre-flop rows held in memory and shuffled together", default = 5000000)
    parser.add_argument("--train_batch_size", type = int, help = "rows per training step", default = 4000)
    parser.add_argument("--epochs", type = int, help = "dataset iterations", default = 3)
    parser.add_argument("--threads", type = int, help = "torch threads", default = 2) # 2 seems to be safe on the VMs
    parser.add_argument("--seed", type = int, help = "randomization seed", default = 0)
    parser.add_argument("--player_dim", type = int, help = "player embedding size, 0 disables player IDs", default = 16)

    return parser.parse_args()


def is_train_row(hand_ids):
    """ Checks if we are looking at a training row or not """
    return hand_ids % 10 != 0


def make_hand_index(columns) -> torch.Tensor:
    """ Turns the two hole cards into one integer ID for the hand embedding """

    # Cards are numbered 0..51 as: rank * 4 + suit
    hole1 = columns["hole1"].long()
    hole2 = columns["hole2"].long()
    rank1 = hole1 // 4
    rank2 = hole2 // 4

    # sort so the order of the two cards doesn't matter (A-K and K-A are the same hand)
    high = torch.maximum(rank1, rank2)
    low = torch.minimum(rank1, rank2)
    suited = ((hole1 % 4) == (hole2 % 4)).long()

    index = high * 26 + low * 2 + suited

    # hidden cards are stored as -1, and use the special ID 338
    return torch.where(hole1 >= 0, index, torch.full_like(index, 338))


def make_features(columns):
    """ Raw decision columns -> (rows, 12) float tensor """

    # in-house ref for card encodings:
    #   rank * 4 + suit
    #   - ranks 2 -> A as 0 -> 12 and suits c d h s as 0 -> 3.
    #       2c = 0  * 4 + 0 = 0
    #       2s = 0  * 4 + 3 = 3
    #       8d = 6  * 4 + 1 = 25
    #       As = 12 * 4 + 3 = 51

    hole1 = columns["hole1"]
    hole2 = columns["hole2"]
    known = hole1 >= 0

    # flooring gives us ranks; e.g. aces // 4 = 12
    rank1 = hole1 // 4
    rank2 = hole2 // 4

    big_blind = columns["big_blind"]

    # NOTE: organizing with dim = 1, rotates the decisions rows to append horizontally in the vector space
    return torch.stack([
        # card features; * known seeds 0s when the cards are hidden
        torch.maximum(rank1, rank2) / 12 * known, # always seed high card first
        torch.minimum(rank1, rank2) / 12 * known,
        ((hole1 % 4) == (hole2 % 4)) * known, # suited flag
        (rank1 == rank2) * known, # pair flag

        known, # flag
        # all divided by a scaling factor to get them between 0-1
        # should probably resolve this instead of going off the current dataset maximums
        columns["seat"] / 9,
        columns["n_players"] / 10,
        columns["n_active"] / 10,
        columns["n_raises"],

        # dividing to get the chip values as a relation to the big blind; idea from Cole
        # log1p is essentially "safe log" for the 0 case; using log to prevent the inevitable squish the same divising factor logic from before would provide
        #   since the vast majority of values in this dataset are less than the big blind, outliers would result in strong biases'
        torch.log1p(columns["pot"] / big_blind),
        torch.log1p(columns["to_call"] / big_blind),
        torch.log1p(columns["stack"] / big_blind),
    ], dim = 1)


##### classes ####################

class PokerDataset:
    """ Streams pre-flop decisions from the parquet table and owns the player ID vocabulary """

    # pylint: disable=no-member; pylint has issues finding many of the Arrow compute c++ bindings
    # pre-flop only, and validate potentially bad rows
    _ROW_FILTERS = (
        (pc.field("street") == 0) &
        pc.is_finite(pc.field("big_blind")) & (pc.field("big_blind") > 0) &
        pc.is_finite(pc.field("pot")) & (pc.field("pot") >= 0) &
        pc.is_finite(pc.field("to_call")) & (pc.field("to_call") >= 0) &
        pc.is_finite(pc.field("stack")) & (pc.field("stack") >= 0)
    )
    # pylint: enable=no-member

    def __init__(self, data_path, max_rows = None, section_rows = 5000000, seed = 0, enable_players = True):
        self.data_path = data_path
        self.max_rows = max_rows
        self.section_rows = section_rows
        self.seed = seed
        self.player_lookup = self._build_player_lookup() if enable_players else None

        if enable_players:
            print(f"# training players: {len(self.player_lookup):,}")

    def stream_batches(self, batch_size = 65536):
        """ Yields pre-flop batches as column -> tensor dicts, with players encoded when the lookup exists """

        for table in self._stream_tables(batch_size, read_players = self.player_lookup is not None):
            # Convert the current batch to PyTorch tensors
            columns = {
                column: torch.tensor(table[column].to_numpy()) for column in COLUMNS
            }

            if self.player_lookup is not None:
                # pylint: disable=no-member
                player_id = pc.add(
                    pc.fill_null(
                        # sub 1 for the not found case, allowing these to be 0 and the actual 0 case to become non-zero ("pre-existing")
                        pc.index_in(table["player_id"], value_set = self.player_lookup), -1
                    ), 1
                )
                # pylint: enable=no-member

                columns["player_id"] = torch.tensor(player_id.to_numpy())

            # Send the batch to the training code.
            yield columns


    def _stream_tables(self, batch_size = 65536, read_players = True):
        """ Reads pre-flop data in batches and yields filtered Arrow tables """

        n_rows = 0
        files = sorted(self.data_path.rglob("*.parquet"))

        # shuffling the original dataset organization so they aren't grouped by stake
        random.Random(self.seed).shuffle(files)

        for path in files:
            dataset = ds.dataset(path, format = "parquet")
            scanner = dataset.scanner(
                columns = COLUMNS + ["player_id"] if read_players else COLUMNS,
                filter = self._ROW_FILTERS,
                batch_size = batch_size
            )

            for batch in scanner.to_batches():
                # Skip batches that contain no valid pre-flop rows
                if batch.num_rows == 0:
                    continue

                table = pa.Table.from_batches([batch])

                if self.max_rows is not None:
                    remaining = self.max_rows - n_rows

                    if remaining <= 0:
                        return

                    table = table.slice(0, remaining)

                # Count only the rows we're actually yielding.
                n_rows += table.num_rows

                yield table

                # Stop once the requested row limit is reached.
                if self.max_rows is not None and n_rows >= self.max_rows:
                    return


    def _build_player_lookup(self):
        """ Sorted Arrow array of player IDs seen in training rows only, so test players can't leak into the vocabulary """

        players = set()
        for table in self._stream_tables():
            train_subset = pa.array(is_train_row(table["hand_id"].to_numpy()))
            players.update(pc.unique(table["player_id"].filter(train_subset)).to_pylist()) #pylint: disable=no-member

        players.discard(None)

        return pa.array(sorted(players), type = pa.string())


    def stream_sections(self):
        """
        Groups streamed batches into larger sections for better randomization in training.
        Inobvious here as to why, but it's relevant for the training `randperm()` call.
        """

        batches = []
        n_rows = 0

        for columns in self.stream_batches():
            batches.append(columns)
            n_rows += len(columns["action"])

            if n_rows >= self.section_rows:
                yield self._merge_section(batches)
                batches = []
                n_rows = 0

        # whatever is left at the end of the data is a smaller final section, or the
        if batches:
            yield self._merge_section(batches)


    @staticmethod
    def _merge_section(batches):
        """ Concatenates a list of column -> tensor batches into a single column -> tensor section """
        resolved_section = {}
        for column in batches[0]:
            column_batches = []
            for batch in batches:
                column_batches.append(batch[column])

            resolved_section[column] = torch.cat(column_batches)

        return resolved_section


    @property
    def n_players(self):
        """ player embedding size """
        return 0 if self.player_lookup is None else len(self.player_lookup) + 1


class PreflopNet(torch.nn.Module):
    """ Feed-forward network over the row features and learned hand and player embeddings """

    def __init__(self, n_players, n_hands = 339, hand_dim = 8, player_dim = 16):
        super().__init__()
        self.hand_embedding = torch.nn.Embedding(n_hands, hand_dim)
        self.player_embedding = torch.nn.Embedding(n_players, player_dim, sparse = True) if player_dim > 0 else None

        # Sequential for linear cxts - https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sequential.html
        # Why LeakyReLU? - https://medium.com/@rajamails19/from-relu-to-gelu-the-pytorch-activation-handbook-5579b682a3c9
        # - Lightweight non-linearity
        # - Helps the dead neuron case: ~9% with ReLU vs ~2% with LeakyReLU
        #    - Accuracy was effectively unchanged, so this is essentially a cheap safeguard rather than a performance gain
        # Why not an alternative like GELU or SiLU?
        # - Neither showed a measurable benefit in this net (2M rows, 3 epochs, 3 seeds), and both trained slightly slower
        #    - GELU is commonly used in deeper architectures like transformers. This model may move to one if it's extended to predict
        #          later streets (not just the pre-flop), where action sequences matter.
        #    - SiLU is commonly used in deep CNNs, where smoother gradients help more. It also had
        #          slightly worse test loss here.

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(len(COLUMNS) + hand_dim + player_dim, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(64, len(ACTION_NAMES)),
        )


    def forward(self, features, hands, players = None):
        """ Concatenates each row's hand embedding, and player embedding if enabled, onto its features """
        inputs = [features, self.hand_embedding(hands)]
        if self.player_embedding is not None:
            inputs.append(self.player_embedding(players.long()))

        return self.mlp(torch.cat(inputs, dim = 1))


class PreflopTraining:
    """ Runs the training loop for a model over a dataset's training rows """

    _UNKNOWN_PLAYER_RATE = 0.01 # share of training samples treated as new/unknown players

    def __init__(self, model, dataset, batch_size = 4000):
        self.model = model
        self.dataset: PokerDataset = dataset
        self.batch_size: int = batch_size
        self.optimizers: list = self._setup_optimizers()


    def _setup_optimizers(self):
        """ Adam for the dense network, SparseAdam for the player embedding """

        # Why Adam?
        # - Smoothes out noisy mini-batch gradients
        # - Per-weight adaptive step sizes suit our mix of scaled features and embeddings, and doesn't need tuning
        # - We don't need decay
        dense_parameters = [parameter for name, parameter in self.model.named_parameters() if not name.startswith("player_embedding")]
        optimizers = [torch.optim.Adam(dense_parameters)]

        # Why SparseAdam for the player embedding?
        # - only need to touch the players in the current batch; std Adam updates every player's vector
        # - Adam's cost is per step, so it grows quickly with smaller batch sizes (more steps per epoch)
        #    - Estimated full-vocab epoch: ~10 -> ~4 min at batch size 4000, ~30 -> ~5 min at batch size 1000
        # - players not in a batch no longer drift on leftover momentum
        # Why not use it for everything..?
        # - The rest of the network has dense gradients, which SparseAdam doesn't support
        if self.model.player_embedding is not None:
            optimizers.append(torch.optim.SparseAdam(self.model.player_embedding.parameters()))

        return optimizers


    def run_training(self, epochs):
        """ Trains over every section of the dataset for each epoch """
        for epoch in range(epochs):
            for columns in self.dataset.stream_sections():
                # keeping a subset for training only, inverse logic done later for testing so we can use the same row boundaries
                train_subset = is_train_row(columns["hand_id"])
                if not train_subset.any():
                    return

                train_in = make_features(columns)[train_subset]
                train_players = columns["player_id"][train_subset] if self.dataset.player_lookup is not None else None
                train_hands = make_hand_index(columns)[train_subset]
                train_target = columns["action"][train_subset].long() # cross_entropy() requires long/byte, fails without

                for index in torch.randperm(len(train_target)).split(self.batch_size):
                    player_sample = None
                    if train_players is not None:
                        # establishing a base case for new/unknown players during testing; overriding ~1% of the samples for it
                        player_sample = train_players[index].clone()
                        player_sample[torch.rand(len(train_players[index])) < self._UNKNOWN_PLAYER_RATE] = 0

                    self._take_next_training_step(train_in[index], train_hands[index], player_sample, train_target[index])

            print(f"epoch {epoch + 1}/{epochs}")


    def _take_next_training_step(self, features, hands, players, target):
        """ One forward pass, backprop, and optimizer update """

        # Why cross-entropy? (looking for balanced classifications)
        # - Lowest loss comes from matching the true action probabilities, which suits modeling how humans mix actions
        # - corrects confident mistakes strongly when paired with softmax
        # Why not an alternative like class weights, focal loss, or MSE?
        # - class weights & focal loss handle imbalance by distorting the probabilities we're trying to learn
        # - MSE penalizes confident mistakes less and typically trains slowly on softmax outputs
        loss = torch.nn.functional.cross_entropy(self.model(features, hands, players), target)

        for optimizer in self.optimizers:
            optimizer.zero_grad()

        loss.backward() # backprop

        for optimizer in self.optimizers:
            optimizer.step() # next


@dataclass
class AccuracyTracker:
    """ Running test accuracy overall and split by whether the cards are known """

    # can prepopulate mutable defaults w/ any Callable it seems
    # https://stackoverflow.com/questions/69564830/python-dataclass-setting-default-list-with-values
    hit_counts: dict[str, int] = field(
        default_factory = lambda: {"combined": 0, "cards_known": 0, "cards_unknown": 0}
    )
    total_counts: dict[str, int] = field(
        default_factory = lambda: {"combined": 0, "cards_known": 0, "cards_unknown": 0}
    )


    def update_accuracy_counts(self, n_correct, known):
        """ Adds one batch of results """
        self._update_individual_count("combined", n_correct)
        self._update_individual_count("cards_known", n_correct[known])
        self._update_individual_count("cards_unknown", n_correct[~known])


    def _update_individual_count(self, name, n_correct):
        """ Adds a boolean tensor of per-row results to one group """
        self.hit_counts[name] += n_correct.sum().item()
        self.total_counts[name] += len(n_correct)


    def print_accuracy(self):
        """ Prints the results table """
        print(f"\n{"test results":<14}{"rows":>12}{"accuracy":>10}")
        for name, hit_count in self.hit_counts.items():
            print(f"{name:<14}{self.total_counts[name]:>12,}{hit_count / max(self.total_counts[name], 1):>10.3f}")


class PreflopEvaluator:
    """ Measures a trained model on the dataset's test rows and on sample hands """

    def __init__(self, model, dataset):
        self.model = model
        self.dataset = dataset


    def evaluate_model(self):
        """ Returns an AccuracyTracker over the test rows """

        metrics = AccuracyTracker()

        with torch.no_grad(): # can more or less use no_grad() for all testing since we don't need to do backprop
            for columns in self.dataset.stream_batches():
                test_subset = ~is_train_row(columns["hand_id"])
                if not test_subset.any():
                    continue

                test = {column: values[test_subset] for column, values in columns.items()}
                test_in = make_features(test)
                test_hands = make_hand_index(test)

                predictions = self.model(test_in, test_hands, test.get("player_id")).argmax(dim = 1)
                correct = predictions == test["action"].long()
                known = test["hole1"] >= 0 # -1 indicates unknown

                metrics.update_accuracy_counts(correct, known)

        return metrics


    def print_example_hands(self, sample_hands):
        """ Sanity check/hand samples: first to act in a 6 player game, nobody has raised, 100 big blind stacks """

        if not sample_hands:
            print("No sample hands given, skipping example...")
            return

        n_samples = len(sample_hands)

        # NOTE: .full((n,) ...) looks awkward but .full() expects a tuple or list and we only need 1D
        # https://docs.pytorch.org/docs/2.15/generated/torch.full.html
        columns = {
            "hole1": torch.tensor([hand[0] for hand in sample_hands.values()]),
            "hole2": torch.tensor([hand[1] for hand in sample_hands.values()]),
            "seat": torch.full((n_samples,), 2), # first player after sm/bb
            "n_players": torch.full((n_samples,), 6), # 6 players
            "n_active": torch.full((n_samples,), 6),
            "n_raises": torch.zeros(n_samples), # no raises
            "big_blind": torch.ones(n_samples),
            "pot": torch.full((n_samples,), 1.5), # just small + big blind
            "to_call": torch.ones(n_samples),
            "stack": torch.full((n_samples,), 100.0) # buy in default
        }

        with torch.no_grad():
            # all sample hands belong to the unknown player (index 0)
            probablities = self.model(make_features(columns), make_hand_index(columns), torch.zeros(n_samples, dtype = torch.long)).softmax(dim = 1)

        print("\nSelect hands sample")
        print(f"\t{"hand":<5}" + "".join(f"{action:>8}" for action in ACTION_NAMES))
        for hand, hand_probabilities in zip(sample_hands, probablities):
            print(f"\t{hand:<5}" + "".join(f"{prob:>8.2f}" for prob in hand_probabilities.tolist()))


@timer
def main():
    """ main driver - loads, trains, & reports """

    args = get_args()

    # torch cfg
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    # extracting the dataset
    dataset = PokerDataset(args.data_path, args.max_rows, args.section_rows, args.seed, enable_players = args.player_dim > 0)

    # Assemble the network
    model = PreflopNet(dataset.n_players, player_dim = args.player_dim)

    # train the model
    PreflopTraining(model, dataset, args.train_batch_size).run_training(args.epochs)

    # testing/eval/examples
    evaluator = PreflopEvaluator(model, dataset)
    evaluator.evaluate_model().print_accuracy()
    evaluator.print_example_hands(DEFAULT_SAMPLE_HANDS)


if __name__ == "__main__":
    sys.exit(main())
