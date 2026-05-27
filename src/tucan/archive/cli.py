from tucan.model import SturgeonSubmodel
from tucan.zip_utils import unpack_zip_file

import numpy as np 
import polars as pl 
import pandas as pd

import torch
import yaml
import os
import argparse


def predict(input_file, path_to_model_files, n, output_file, num_samples, file_type):

    if num_samples is None:
        num_samples = 1

    # Load classification system
    #with open(os.path.join(path_to_model_files, 'classification_system.yaml'), 'r') as stream:
    #    classification_file = yaml.safe_load(stream)

    #classes = list(classification_file['decoder']['type'].keys())
    #class_names = list(classification_file['encoder']['type'].keys())
    #classification_sizes = len(classes)

    # Load classification system
    with open(os.path.join(path_to_model_files, 'classification_system.yaml'), 'r') as stream:
        classification_file = yaml.safe_load(stream)

    encoder = classification_file['encoder']['type']   # e.g. {"ClassA": 0, "ClassB": 1, ...}
    decoder = classification_file['decoder']['type']   # e.g. {"0": "ClassA", "1": "ClassB", ...} or {0: "ClassA", 1: "ClassB"}

    # Ensure class names are ordered by the class index used by the model outputs
    try:
        # prefer decoder: index -> name
        idx_name_pairs = sorted(
            ((int(k), v) for k, v in decoder.items()),
            key=lambda kv: kv[0]
        )
        class_names = [name for _, name in idx_name_pairs]
        classification_sizes = len(idx_name_pairs)
    except Exception:
        # fallback: sort encoder by its index values
        class_names = [name for name, idx in sorted(encoder.items(), key=lambda kv: int(kv[1]))]
        classification_sizes = len(class_names)
    
    #print("Class index → name mapping used for columns:")
    #for i, name in enumerate(class_names):
    #    print(f"{i} → {name}")

    # Load probe file
    probe_df = pl.read_csv(
        os.path.join(path_to_model_files, 'probe.bed'), 
        separator='\t'
    )
    in_size = len(probe_df)

    # Load input depending on type
    if file_type == 'csv':
        columns = ['probe_id', 'methylation_call']
        bed_file = pl.read_csv(input_file, separator=',', has_header=False)
        bed_file.columns = columns
    elif file_type == 'bed_carlo':
        bed_file = pl.read_csv(input_file, separator=' ')
    elif file_type == 'bed':
        bed_file = pl.read_csv(input_file, separator='\t')
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Ensure correct dtypes
    bed_file = bed_file.select([
        pl.col('probe_id').cast(pl.Utf8).alias('probe_id'),
        pl.col('methylation_call').cast(pl.Int32, strict=False).alias('methylation_call')
    ])

    # Convert methylation_call 0 -> -1
    bed_file = bed_file.with_columns(
        methylation_call=pl.when(pl.col('methylation_call') == 0)
            .then(pl.lit(-1))
            .otherwise(pl.col('methylation_call'))
    )

    # Join with probes
    nn_input = probe_df.join(
        bed_file, 
        left_on='name', 
        right_on='probe_id', 
        how='left', 
        validate='1:1'
    )

    nn_input = nn_input.with_columns(pl.all().fill_null(0))
    
    device = torch.device("cpu")

    print('-------------------------------')
    print('Loading submodels')
    print('-------------------------------')

    models = []
    for i in range(4):  # Tucan uses 4 submodels
        model = SturgeonSubmodel(
            in_size=in_size, 
            classification_sizes=classification_sizes, 
            activation='silu'
        ).to(device)

        chk = torch.load(
            os.path.join(path_to_model_files, 'checkpoints', f'checkpoint_{i}.pt'), 
            weights_only=False, 
            map_location=device
        )
        #print(chk, flush=True)
        model.load_state_dict(chk['model_state'])
        model.eval()
        models.append(model)

            # --- inspect what's inside the checkpoint ---
    # Many checkpoints store params under 'model_state' or 'state_dict'.
    #    if isinstance(chk, dict):
    #        if "model_state" in chk:
    #            state = chk["model_state"]
    #        elif "state_dict" in chk:
    #            state = chk["state_dict"]
    #        else:
    #        # Sometimes the checkpoint *is* the state_dict already
    #            state = chk
    #    else:
    #        state = chk

     #   print(f"\n=== Checkpoint {i} param shapes ===")
      #  for key, value in state.items():
      #      if hasattr(value, "shape"):
      #          print(f"{key}: {tuple(value.shape)}")
      #      else:
            # Non-tensor entries (rare for state_dicts)
       #         print(f"{key}: {type(value).__name__}")

    nn_input = torch.tensor(nn_input['methylation_call'].to_numpy())
    result = np.zeros((num_samples, classification_sizes))

    print('-------------------------------')
    print('Running subsample predictions')
    print('-------------------------------')

    for i in range(num_samples):
        # Find positions with non-zero values
        positions = torch.nonzero(nn_input)

        if n is None:
            if i == 0:
                print('-------------------------------')
                print('Using all positions')
                print('-------------------------------')
            n = len(positions)
        else:
            n = int(n)
            if n > len(positions):
                if i == 0:
                    print('-------------------------------')
                    print('Using all positions')
                    print('-------------------------------')
                n = len(positions)

        random_ind = torch.randperm(
            len(positions), 
            generator=torch.Generator().manual_seed(i)
        )[:n]

        num_feature_input = int(random_ind.shape[0])
        selected_positions = positions[random_ind]

        new_tensor = torch.zeros_like(nn_input)
        new_tensor[selected_positions] = nn_input[selected_positions]
        new_tensor = new_tensor.reshape(1, -1).to(device).to(torch.float32)
        
        outputs = None
        for k in range(4):  # average over 4 submodels
            if outputs is None:
                outputs = models[k](new_tensor)['y']
            else:
                outputs += models[k](new_tensor)['y']
        outputs = outputs / 4

        outputs = torch.nn.Softmax(dim=1)(outputs)
        result[i, :] = outputs.cpu().detach().numpy()

    df = pd.DataFrame(result, columns=class_names)
    df['probes'] = len(df) * [n]

    df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input_file", help="path to input file")
    parser.add_argument("-m", "--model", help="specify path to model zip you want to use.")
    parser.add_argument("-c", "--num_CpGs", help="number of CpG sites (default all available).")
    parser.add_argument("-o", "--output_file", help="path to output file")
    parser.add_argument("-s", "--num_samplings", help="number of random samples of size num_CpGs. Default 1.")
    parser.add_argument("-f", "--file_type", help="input file type 'bed' or 'csv'")

    args = parser.parse_args()
    
    num_samples = 1 if args.num_samplings is None else int(args.num_samplings)
    
    print('---------------------------------------------------------------------')
    print('Running intraoperative methylation based classification')
    print('---------------------------------------------------------------------')

    path_to_model_files = unpack_zip_file(args.model)

    predict(
        args.input_file, 
        path_to_model_files, 
        args.num_CpGs, 
        args.output_file, 
        num_samples, 
        args.file_type
    )


def entrypoint():
    try:
        main()
    except Exception as e:
        import traceback
        print("\n❌ ERROR: An exception occurred in tucan CLI\n")
        traceback.print_exc()
        exit(1)
