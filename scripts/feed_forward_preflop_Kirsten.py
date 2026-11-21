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
# Infer hidden cards based on current representation distributions and what action the player took for better training quality.
#     most significantly, need to unroll the unknown's fold rates back into the weaker hands
#     (Kristian)
#
# Use an alternative hand encoding that better represents how strong the hands are sensitive to how people play them.
#     lets the model learn card strengths based on how people play rather than the mathematical significance
#     (Surja)
#
# Use all of the data, stream the rows instead of loading it all into memory.
#     should make this optional, so we can test with a smaller sample to get the other features implemented first
#     (Cole)
#
# Include player IDs into the decisions/ table and model learning
#     think this would be useful as a PoC that it can adapt to new players/data with non-traditional behavior
#     (Kirsten)
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
import pyarrow.parquet as pq
import pyarrow.dataset as ds ##added by Kirsten
#import time #added by Kirsten testing only

##### constants ####################
ACTION_NAMES = ["fold", "call", "raise"]
COLUMNS = ["hand_id", "seat", "n_players", "n_active", "big_blind", "pot", "to_call", "stack", "n_raises", "hole1", "hole2", "action"]

# pre-flop only, and skip the bad rows: some sites record unknown stacks as inf, and negative blind entries make the pot negative
# TODO: fix or remove these bad rows...
#     (Cole)
ROW_FILTERS = (pc.field("street") == 0) & pc.is_finite(pc.field("stack")) & (pc.field("pot") >= 0)

##### funcs ####################
def get_args():
    """ Argparser retrieval """

    parser = ArgumentParser(description=__doc__)

    parser.add_argument("--data_path", type = Path, help = "path to the decisions parquet table", default = "parquet/decisions")
    parser.add_argument("--max_rows", type = int, help = "pre-flop rows to load", default = 10000000) # TODO: change to optional for final model (Cole)
    parser.add_argument("--epochs", type = int, help = "dataset iterations", default = 3)
    parser.add_argument("--threads", type = int, help = "torch threads", default = 2) # 2 seems to be safe on the VMs

    return parser.parse_args()


##### def load preflop changed by Kirsten to load ordered parts using batches ####################
def load_preflop(data_path, max_rows):
    """ Reads pre-flop rows from randomly ordered parts using batches """
    # TODO: fairly sure there's a better way to do this leveraging pyarrow, to research...
    #     (Kirsten)
    ##used a batch loading approach with pyarrow.dataset using pytorch tensors as output similar to the approach used in my bayesian individual
    #project submisssion which used a pandas dataframe output. Was unable to prove that 
    ##this approach was faster than our original pyarrow.parquet approach

    batches = []
    n_rows = 0

    files = sorted(data_path.rglob("*.parquet"))

    #Preserve the original randomized order (Kirsten)
    # shuffling the original dataset organization so they aren't grouped by stake
    # should remove the fixed seed if we don't want to change other configurations like Adam/ReLU
    random.Random(0).shuffle(files)

    for path in files:
        dataset = ds.dataset(path, format = "parquet")
        
        scanner = dataset.scanner(
            columns = COLUMNS + ["player_id"],#Added by Kirsten 
            filter = ROW_FILTERS,
            batch_size = 65536
        )

        for batch in scanner.to_batches():
            if batch.num_rows == 0:
                continue

            remaining = max_rows - n_rows

            #Do not keep more rows than requested
            if batch.num_rows > remaining:
                batch = batch.slice(0,remaining)

            batches.append(batch)
            n_rows += batch.num_rows # num rows in current table
            
            if n_rows >= max_rows:
                break

        if n_rows >= max_rows:
            break 

    if not batches:
        raise ValueError("No matching pre-flop rows found.")

    table = pa.Table.from_batches(batches)

    result = { 
        column: torch.from_numpy(table[column].to_numpy().copy())
        for column in COLUMNS
    }
    #Preserve player IDs as strings for later encoding
    result["player_id"] = table["player_id"].to_pylist()

    return result

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
        # TODO: should test other normalization strategies to see if those make a significant difference
        #     (Kristian - deprioritized)
        torch.log1p(columns["pot"] / big_blind),
        torch.log1p(columns["to_call"] / big_blind),
        torch.log1p(columns["stack"] / big_blind),
    ], dim = 1)


#Kirsten - Add players to neural network


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
        #probablities = model(make_features(columns)).softmax(dim = 1)commented out by Kirsten
        sample_features = make_features(columns)
##################################################################################################################added by Kirsten
        #make space for player indicators, creates columns initially filled w zeros
        #the number of columnsd depends on how many players the model recognizes
        sample_players = torch.zeros(
        (n, model[0].in_features - sample_features.shape[1])
        )
        #Mark all example hands as belonging to an unknown player (index 0)
        sample_players[:, 0] = 1  # Unknown player

        #combine the poker features with with player id info
        #predict probabilities of folding, calling, raising
        probablities = model(
            torch.cat((sample_features, sample_players), dim=1)
        ).softmax(dim=1)
    #######################################################################################
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

    # Benchmark Kirsten's PyArrow batch loader FIRST
    #code between hashtags beginning at start_time is chatgpt generated for testing the loading time of files only and
    #would not be any part of final code. 
    ##########################################################################################################
#    """ 
#    start_time = time.perf_counter()

#     columns = load_preflop(args.data_path, args.max_rows)

#     load_time = time.perf_counter() - start_time
#     print(f"PyArrow batch loading time: {load_time:.3f} seconds")

#     # Benchmark original loader SECOND
#     import feed_forward_preflop as original

#     start_original = time.perf_counter()

#     original_columns = original.load_preflop(
#         args.data_path, args.max_rows
#     )

#     original_time = time.perf_counter() - start_original
#     print(f"Original loading time: {original_time:.3f} seconds")

#     # Verify both loaders produce identical data
#     print("Identical data:", all(
#         torch.equal(columns[c], original_columns[c])
#         for c in COLUMNS
#     ))"""
###########################################################################################################
    # Kirsten - Load pre-flop decisions using PyArrow batches
    columns = load_preflop(args.data_path, args.max_rows)

    features = make_features(columns)

    labels = columns["action"].long() # cross_entropy() requires long/byte, fails without
    known = columns["hole1"] >= 0 # -1 indicates unknown

    # splitting test/training subsets, just pulling about 1 in 10 for testing
    test_subset = columns["hand_id"] % 10 == 0
#######################################################################################################
    # Kirsten - Build player vocabulary from training data only
    player_ids = columns["player_id"]

    training_players = {
        player
        for player, is_test in zip(player_ids, test_subset.tolist())
        if not is_test and player is not None
    }

    player_to_index = {
        player: index + 1
        for index, player in enumerate(sorted(training_players))
    }

    # Unknown or unseen players receive index 0
    player_indices = torch.tensor([
        player_to_index.get(player, 0)
        for player in player_ids
    ], dtype=torch.long)

    print("Training players:", len(player_to_index))
    print("Player indices:", player_indices[:10])
    print("Unseen test decisions:", (player_indices[test_subset] == 0).sum().item())
#################################################################################################################
    
    #Kirsten - Add player IDs to model input features
    #one_hot()converts player index into indicator values instead of numerical, int indices to vectors of zeros and ones
    #torch.nn.functional PyTorch's neural network functions
    player_one_hot = torch.nn.functional.one_hot(
        player_indices,##one integer per poker decision
        num_classes = len(player_to_index)+1
    ).float()

    #player_one_hot = torch.zeros_like(player_one_hot)#disable player id's for testing accuracy comparison it performs better w/Id's
    #on 1000 rows and 3 epochs
    features = torch.cat((features, player_one_hot),dim=1)

    train_in, train_target = features[~test_subset], labels[~test_subset]
    test_in, test_target = features[test_subset], labels[test_subset]
    

    #Kirsten - split player ids with their decisions
    train_player = player_indices[~test_subset]
    test_player = player_indices[test_subset]
    print("Training player indices:", train_player.shape)
    print("Testing player indices:", test_player.shape)


    print(f"Loaded {len(labels):,} pre-flop rows ({len(train_target):,} train, {len(test_target):,} test)")

    # Sequential for linear cxts - https://docs.pytorch.org/docs/2.15/generated/torch.nn.Sequential.html
    model = torch.nn.Sequential(
        torch.nn.Linear(features.shape[1], 64),
        torch.nn.ReLU(), # pretty much just using the most common features (which the baove source uses), ReLU happens to be suitable for this application
        torch.nn.Linear(64, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, len(ACTION_NAMES)),
    )

    # Tuning handler, Adam appears to just be the most common optimizer, can tune if noted to be a limiting factor
    # default lr is already 1e-3 (from lib: `float | Tensor = 1e-3`), specified it because we may want to tune this
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

    for _ in range(args.epochs):
        # further shuffling to mostly isolate the individual games into different batches
        # TODO: re-evaluate how many steps to split into once we ingest the full dataset, probably needs to be a function of N
        #     (Cole)
        for index in torch.randperm(len(train_target)).split(4000):
            loss = torch.nn.functional.cross_entropy(model(train_in[index]), train_target[index])
            optimizer.zero_grad()
            loss.backward() # backprop
            optimizer.step()

    # run a test to git the hit rate for accuracy
    with torch.no_grad(): # can more or less use no_grad() for all testing since we don't need to do backprop
        correct = model(test_in).argmax(dim = 1) == test_target

    print(f"\n{"test results":<14}{"rows":>12}{"accuracy":>10}")
    print(f"{"combined":<14}{len(correct):>12,}{correct.float().mean().item():>10.3f}")
    for name, group in [("cards known", known), ("cards unknown", ~known)]:
        test_group = group[test_subset]
        print(f"{name:<14}{test_group.sum().item():>12,}{correct[test_group].float().mean().item():>10.3f}")

    print_example_hands(model)


if __name__ == "__main__":
    sys.exit(main())
