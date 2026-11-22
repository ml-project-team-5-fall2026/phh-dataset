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

##### other TODOs ####################
# TODO: not attached to any one part of the code, just improvements to make:
# Use an alternative hand encoding that better represents how strong the hands are sensitive to how people play them.
#     lets the model learn card strengths based on how people play rather than the mathematical significance
#     (Surja)
#
# More hidden layers & epochs
#     (Surja)
######################################

# std libs
import sys
import random
from pathlib import Path
from argparse import ArgumentParser

# external libs
import torch
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds

##### constants ####################
ACTION_NAMES = ["fold", "call", "raise"]
COLUMNS = ["hand_id", "seat", "n_players", "n_active", "big_blind", "pot", "to_call", "stack", "n_raises", "hole1", "hole2", "action"]

# pylint: disable=no-member; pylint has issues finding many of the c++ bindings
# pre-flop only, and validate potentially bad rows
ROW_FILTERS = (
    (pc.field("street") == 0) &
    pc.is_finite(pc.field("big_blind")) & (pc.field("big_blind") > 0) &
    pc.is_finite(pc.field("pot")) & (pc.field("pot") >= 0) &
    pc.is_finite(pc.field("to_call")) & (pc.field("to_call") >= 0) &
    pc.is_finite(pc.field("stack")) & (pc.field("stack") >= 0)
)
# pylint: enable=no-member

##### funcs ####################
def get_args():
    """ Argparser retrieval """

    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--data_path", type = Path, help = "path to the decisions parquet table", default = Path("parquet/decisions"))
    parser.add_argument("--max_rows", type = int, help = "total pre-flop rows to read", default = None)
    parser.add_argument("--section_rows", type = int, help = "max pre-flop rows held in memory and shuffled together", default = 5000000)
    parser.add_argument("--epochs", type = int, help = "dataset iterations", default = 3)
    parser.add_argument("--threads", type = int, help = "torch threads", default = 2) # 2 seems to be safe on the VMs
    parser.add_argument("--seed", type = int, help = "randomization seed", default = 0)

    return parser.parse_args()

def stream_preflop_tables(data_path, max_rows = None, batch_size = 65536, seed = 0):
    """ Reads pre-flop data in batches and yields filtered Arrow tables """

    n_rows = 0
    files = sorted(data_path.rglob("*.parquet"))

    # shuffling the original dataset organization so they aren't grouped by stake
    random.Random(seed).shuffle(files)

    for path in files:
        dataset = ds.dataset(path, format = "parquet")
        scanner = dataset.scanner(
            columns = COLUMNS + ["player_id"],
            filter = ROW_FILTERS,
            batch_size = batch_size
        )

        for batch in scanner.to_batches():
            # Skip batches that contain no valid pre-flop rows
            if batch.num_rows == 0:
                continue

            table = pa.Table.from_batches([batch])

            if max_rows is not None:
                remaining = max_rows - n_rows

                if remaining <= 0:
                    return

                table = table.slice(0, remaining)

            # Count only the rows we're actually yielding.
            n_rows += table.num_rows

            yield table

            # Stop once the requested row limit is reached.
            if max_rows is not None and n_rows >= max_rows:
                return


def build_player_lookup(data_path, max_rows, seed):
    """ Sorted Arrow array of player IDs seen in training rows only, so test players can't leak into the vocabulary """

    players = set()
    for table in stream_preflop_tables(data_path, max_rows, seed = seed):
        train_subset = pa.array(table["hand_id"].to_numpy() % 10 != 0)
        players.update(pc.unique(table["player_id"].filter(train_subset)).to_pylist()) #pylint: disable=no-member

    players.discard(None)

    return pa.array(sorted(players), type = pa.string())


def stream_preflop(data_path, player_lookup, max_rows = None, batch_size = 65536, seed = 0):
    """ Yields pre-flop batches as column -> tensor dicts, with players encoded by player_lookup """

    for table in stream_preflop_tables(data_path, max_rows, batch_size, seed):
        # Convert the current batch to PyTorch tensors
        columns = {
            column: torch.tensor(table[column].to_numpy()) for column in COLUMNS
        }

        # index_in() is null for players outside the vocabulary; shifting by 1 makes those index 0 (unknown)
        # pylint: disable=no-member
        player_id = pc.add(
            pc.fill_null(
                pc.index_in(table["player_id"], value_set = player_lookup), -1
            ), 1
        )
        # pylint: enable=no-member

        columns["player_id"] = torch.tensor(player_id.to_numpy())

        # Send the batch to the training code.
        yield columns


def walk_training_dataset(data_path, player_lookup, max_rows, section_rows, seed):
    """
    Groups streamed batches into larger sections for better randomization in training.
    Inobvious here as to why, but it's relevant for the training `randperm()` call.
    """

    batches = []
    n_rows = 0

    for columns in stream_preflop(data_path, player_lookup, max_rows, seed = seed):
        batches.append(columns)
        n_rows += len(columns["action"])

        if n_rows >= section_rows:
            yield merge_section(batches)
            batches = []
            n_rows = 0

    # whatever is left at the end of the data is a smaller final section, or the
    if batches:
        yield merge_section(batches)


def merge_section(batches):
    """ Concatenates a list of column -> tensor batches into a single column -> tensor section """
    resolved_section = {}
    for column in COLUMNS + ["player_id"]:
        column_batches = []
        for batch in batches:
            column_batches.append(batch[column])

        resolved_section[column] = torch.cat(column_batches)

    return resolved_section


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


def add_player_features(features, player_id, n_players):
    """ concats one-hot player columns """

    # TODO: One-hot doesn't work with the full dataset, probably need a torch Embedding
    player_one_hot = torch.nn.functional.one_hot(player_id.long(), num_classes = n_players).float() # pylint: disable=not-callable
    return torch.cat((features, player_one_hot), dim = 1)


def print_example_hands(model):
    """ Sanity check/hand samples: first to act in a 6 player game, nobody has raised, 100 big blind stacks """

    # put high card to the left, not resolving high card like in make_features
    sample_hands = {
        "AA": (51, 50),
        "AKs": (51, 47),
        "JTs": (39, 35),
        "72o": (20, 1),
        "??": (-1, -1)
    }

    n = len(sample_hands)

    # NOTE: .full((n,) ...) looks awkward but full expects a tuple or list and we only need 1D
    # https://docs.pytorch.org/docs/2.15/generated/torch.full.html
    columns = {
        "hole1": torch.tensor([hand[0] for hand in sample_hands.values()]),
        "hole2": torch.tensor([hand[1] for hand in sample_hands.values()]),
        "seat": torch.full((n,), 2), # first player after sm/bb
        "n_players": torch.full((n,), 6), # 6 players
        "n_active": torch.full((n,), 6),
        "n_raises": torch.zeros(n), # no raises
        "big_blind": torch.ones(n),
        "pot": torch.full((n,), 1.5), # just small + big blind
        "to_call": torch.ones(n),
        "stack": torch.full((n,), 100.0) # buy in default
    }

    with torch.no_grad():
        # all sample hands belong to the unknown player (index 0)
        sample_features = make_features(columns)
        n_players = model[0].in_features - sample_features.shape[1]
        probablities = model(add_player_features(sample_features, torch.zeros(n), n_players)).softmax(dim = 1)

    print("\nSelect hands sample")
    print(f"\t{"hand":<5}" + "".join(f"{action:>8}" for action in ACTION_NAMES))
    for hand, hand_probabilities in zip(sample_hands, probablities):
        print(f"\t{hand:<5}" + "".join(f"{prob:>8.2f}" for prob in hand_probabilities.tolist()))


def main():
    """ main driver - loads, trains, & reports """
    # TODO: convert most of this to a proper class structure, this isn't great
    #     (Kristian)

    args = get_args()
    torch.manual_seed(0) # fixed for now, could probably be an arg
    torch.set_num_threads(args.threads)

    player_lookup = build_player_lookup(args.data_path, args.max_rows, args.seed)
    print(f"# training players: {len(player_lookup):,}")

    # + 1 reserves index 0 for unknown/unseen players
    n_players = len(player_lookup) + 1

    # Sequential for linear cxts - https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sequential.html
    model = torch.nn.Sequential(
        torch.nn.Linear(len(COLUMNS) + n_players, 64),
        torch.nn.ReLU(), # pretty much just using the most common features (which the above source uses), ReLU happens to be suitable for this application
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, len(ACTION_NAMES)),
    )

    # Tuning handler, Adam appears to just be the most common optimizer, can tune if noted to be a limiting factor
    # default lr is already 1e-3 (from lib: `float | Tensor = 1e-3`), specified it because we may want to tune this
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for epoch in range(args.epochs):
        for columns in walk_training_dataset(args.data_path, player_lookup, args.max_rows, args.section_rows, args.seed):
            # keeping a subset for training only, inverse logic done later for testing so we can use the same row boundaries
            train_subset = columns["hand_id"] % 10 != 0
            if not train_subset.any():
                continue

            train_in = make_features(columns)[train_subset]
            train_players = columns["player_id"][train_subset]
            train_target = columns["action"][train_subset].long() # cross_entropy() requires long/byte, fails without

            # TODO: re-evaluate how many steps to split into once we ingest the full dataset, probably needs to be a function of N
            #     (Cole)
            for index in torch.randperm(len(train_target)).split(4000):
                batch_in = add_player_features(train_in[index], train_players[index], n_players)
                loss = torch.nn.functional.cross_entropy(model(batch_in), train_target[index])
                optimizer.zero_grad()
                loss.backward() # backprop
                optimizer.step()

        print(f"epoch {epoch + 1}/{args.epochs}")


    hit_counts = {"combined": 0, "cards_known": 0, "cards_unknown": 0}
    total_counts = hit_counts.copy()

    with torch.no_grad(): # can more or less use no_grad() for all testing since we don't need to do backprop
        for columns in stream_preflop(args.data_path, player_lookup, args.max_rows, seed = args.seed):
            test_subset = columns["hand_id"] % 10 == 0
            if not test_subset.any():
                continue

            test = {column: values[test_subset] for column, values in columns.items()}
            test_in = make_features(test)

            predictions = torch.cat([
                model(add_player_features(test_in[index], test["player_id"][index], n_players)).argmax(dim = 1)
                for index in torch.arange(len(test_in)).split(4000)
            ])

            correct = predictions == test["action"].long()
            known = test["hole1"] >= 0 # -1 indicates unknown

            # metrics
            hit_counts["combined"] += correct.sum().item()
            total_counts["combined"] += len(correct)

            hit_counts["cards_known"] += correct[known].sum().item()
            total_counts["cards_known"] += known.sum().item()

            hit_counts["cards_unknown"] += correct[~known].sum().item()
            total_counts["cards_unknown"] += (~known).sum().item()

    print(f"\n{"test results":<14}{"rows":>12}{"accuracy":>10}")
    for name, hit_count in hit_counts.items():
        print(f"{name:<14}{total_counts[name]:>12,}{hit_count / max(total_counts[name], 1):>10.3f}")

    print_example_hands(model)


if __name__ == "__main__":
    sys.exit(main())
