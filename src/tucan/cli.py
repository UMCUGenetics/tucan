import argparse
import zipfile
import tempfile
import os
import shutil

import numpy as np
import pandas as pd
import polars as pl
import yaml
import torch

from tucan.model import SturgeonSubmodel


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------
def predict(
    input_file,
    model_zip_path,
    n,
    output_file,
    num_samples,
    file_type,
):
    if num_samples is None:
        num_samples = 1

    # --------------------------------------------------------------
    # Extract model zip to temp directory
    # --------------------------------------------------------------
    tmpdir = tempfile.mkdtemp(prefix="tucan_model_")

    try:
        with zipfile.ZipFile(model_zip_path, "r") as zf:
            zf.extractall(tmpdir)

        # --------------------------------------------------------------
        # Adjust this if your zip contains a top-level folder
        # e.g. tmpdir/cns-v2/
        # --------------------------------------------------------------
        model_root = tmpdir

        # Example if needed:
        # model_root = os.path.join(tmpdir, "cns-v2")

        # --------------------------------------------------------------
        # Load classification config
        # --------------------------------------------------------------
        with open(
            os.path.join(model_root, "classification_system.yaml"),
            "r",
        ) as f:
            classification_file = yaml.safe_load(f)

        encoder = classification_file["encoder"]["type"]
        decoder = classification_file["decoder"]["type"]

        # --------------------------------------------------------------
        # Robust class ordering
        # --------------------------------------------------------------
        try:
            idx_name_pairs = sorted(
                ((int(k), v) for k, v in decoder.items()),
                key=lambda kv: kv[0],
            )

            class_names = [name for _, name in idx_name_pairs]
            classification_sizes = len(idx_name_pairs)

        except Exception:
            class_names = [
                name
                for name, idx in sorted(
                    encoder.items(),
                    key=lambda kv: int(kv[1]),
                )
            ]

            classification_sizes = len(class_names)

        # --------------------------------------------------------------
        # Load probe file
        # --------------------------------------------------------------
        probe_df = pl.read_csv(
            os.path.join(model_root, "probe.bed"),
            separator="\t",
        )

        in_size = len(probe_df)

        # --------------------------------------------------------------
        # Load input methylation data
        # --------------------------------------------------------------
        if file_type == "csv":
            bed_file = pl.read_csv(
                input_file,
                separator=",",
                has_header=False,
                new_columns=["probe_id", "methylation_call"],
            )

        elif file_type == "bed":
            bed_file = pl.read_csv(
                input_file,
                separator="\t",
            )

        elif file_type == "bed_carlo":
            bed_file = pl.read_csv(
                input_file,
                separator=" ",
            )

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

        # --------------------------------------------------------------
        # Ensure correct dtypes
        # --------------------------------------------------------------
        bed_file = bed_file.select(
            [
                pl.col("probe_id").cast(pl.Utf8),

                pl.col("methylation_call")
                .cast(pl.Int32, strict=False)
                .alias("methylation_call"),
            ]
        )

        # --------------------------------------------------------------
        # Convert 0 -> -1
        # --------------------------------------------------------------
        bed_file = bed_file.with_columns(
            methylation_call=(
                pl.when(pl.col("methylation_call") == 0)
                .then(-1)
                .otherwise(pl.col("methylation_call"))
            )
        )

        # --------------------------------------------------------------
        # Join probes with input
        # --------------------------------------------------------------
        nn_input = probe_df.join(
            bed_file,
            left_on="name",
            right_on="probe_id",
            how="left",
            validate="1:1",
        ).with_columns(pl.all().fill_null(0))

        nn_input = torch.tensor(
            nn_input["methylation_call"].to_numpy(),
            dtype=torch.float32,
        )

        # --------------------------------------------------------------
        # Device
        # --------------------------------------------------------------
        device = torch.device("cpu")

        # --------------------------------------------------------------
        # Load Tucan submodels
        # --------------------------------------------------------------
        print("-------------------------------")
        print("Loading Tucan submodels")
        print("-------------------------------")

        models = []

        for i in range(4):

            model = SturgeonSubmodel(
                in_size=in_size,
                classification_sizes=classification_sizes,
                activation="silu",
            ).to(device)

            chk = torch.load(
                os.path.join(
                    model_root,
                    "checkpoints",
                    f"checkpoint_{i}.pt",
                ),
                weights_only=False,
                map_location=device,
            )

            model.load_state_dict(chk["model_state"])
            model.eval()

            models.append(model)

        # --------------------------------------------------------------
        # Run predictions
        # --------------------------------------------------------------
        print("-------------------------------")
        print("Running subsample predictions")
        print("-------------------------------")

        result = np.zeros((num_samples, classification_sizes))

        positions = torch.nonzero(nn_input).flatten()

        for i in range(num_samples):

            # ----------------------------------------------------------
            # Determine number of CpGs
            # ----------------------------------------------------------
            if n is None or int(n) > len(positions):

                if i == 0:
                    print("-------------------------------")
                    print("Using all positions")
                    print("-------------------------------")

                n_used = len(positions)

            else:
                n_used = int(n)

            # ----------------------------------------------------------
            # Random subsampling
            # ----------------------------------------------------------
            random_ind = torch.randperm(
                len(positions),
                generator=torch.Generator().manual_seed(i),
            )[:n_used]

            selected_positions = positions[random_ind]

            # ----------------------------------------------------------
            # Create sparse input tensor
            # ----------------------------------------------------------
            new_tensor = torch.zeros_like(nn_input)

            new_tensor[selected_positions] = nn_input[selected_positions]

            new_tensor = (
                new_tensor
                .reshape(1, -1)
                .to(device)
                .to(torch.float32)
            )

            # ----------------------------------------------------------
            # Ensemble prediction
            # ----------------------------------------------------------
            outputs = None

            for k in range(4):

                pred = models[k](new_tensor)["y"]

                if outputs is None:
                    outputs = pred
                else:
                    outputs += pred

            outputs = outputs / 4

            outputs = torch.nn.functional.softmax(
                outputs,
                dim=1,
            )

            result[i, :] = (
                outputs
                .cpu()
                .detach()
                .numpy()[0]
            )

        # --------------------------------------------------------------
        # Write output
        # --------------------------------------------------------------
        df = pd.DataFrame(
            result,
            columns=class_names,
        )

        df["probes"] = n_used

        df.to_csv(
            output_file,
            index=False,
        )

    finally:
        # --------------------------------------------------------------
        # Cleanup temp directory AFTER inference
        # --------------------------------------------------------------
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-c", "--num_CpGs")
    parser.add_argument("-o", "--output_file", required=True)
    parser.add_argument("-s", "--num_samplings")
    parser.add_argument("-f", "--file_type", required=True)

    args = parser.parse_args()

    num_samples = (
        int(args.num_samplings)
        if args.num_samplings
        else 1
    )

    print("---------------------------------------------------------------------")
    print("Running intraoperative methylation based classification (Tucan)")
    print("---------------------------------------------------------------------")

    predict(
        input_file=args.input_file,
        model_zip_path=args.model,
        n=args.num_CpGs,
        output_file=args.output_file,
        num_samples=num_samples,
        file_type=args.file_type,
    )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def entrypoint():

    try:
        main()

    except Exception:

        import traceback

        print("\n❌ ERROR: An exception occurred in Tucan CLI\n")

        traceback.print_exc()

        exit(1)
