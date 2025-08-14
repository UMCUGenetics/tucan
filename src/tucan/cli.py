from sturgeon_v2.model import SturgeonSubmodel
from sturgeon_v2.zip_utils import unpack_zip_file

import numpy as np 
import polars as pl 
import pandas as pd

import torch
import yaml
import os
import argparse

def predict(input_file, path_to_model_files, n, output_file, num_samples, file_type):

    if num_samples == None:
        num_samples = 1

    with open(os.path.join(path_to_model_files, 'classification_system.yaml'), 'r') as stream:
        classification_file = yaml.safe_load(stream)

    classes = list(classification_file['decoder']['type'].keys())

    class_names = list(classification_file['encoder']['type'].keys())

    output_size = len(classes)

    probe_df = pl.read_csv(
        os.path.join(path_to_model_files, 'probe.bed'), 
        separator='\t'
    )

    input_size = len(probe_df)

    if file_type == 'csv':
        columns = ['probe_id', 'methylation_call']

        bed_file = pl.read_csv(
            input_file,
            separator=',',
            has_header=False
        )

        bed_file.columns = columns
    if file_type == 'bed_carlo':
        bed_file = pl.read_csv(
            input_file,
            separator=' '
        )
    if file_type == 'bed':
        bed_file = pl.read_csv(
            input_file,
            separator='\t'
        )
        
    bed_file = bed_file.with_columns(
        methylation_call=pl.when(pl.col('methylation_call')==0)
                .then(pl.lit(-1))
                .otherwise(pl.col('methylation_call'))
    )


    nn_input = probe_df.join(
        bed_file, 
        left_on='name', 
        right_on='probe_id', 
        how='left', 
        validate='1:1'
    )

    nn_input = nn_input.with_columns(
        pl.all()
        .fill_null(0)
    )
    
    device = torch.device("cpu")

    print('-------------------------------')
    print('Loading submodels')
    print('-------------------------------')

    models = []

    for i in range(4):
        model = SturgeonSubmodel(
            input_size = input_size, 
            output_size = output_size, 
            activation = 'silu' 
        ).to(device)

        chk = torch.load(os.path.join(path_to_model_files, 'checkpoints', f'checkpoint_{i}.pt'), weights_only=False, map_location=device)

        model.load_state_dict(chk)

        model.eval()

        models.append(model)

    nn_input = torch.tensor(nn_input['methylation_call'].to_numpy())

    result = np.zeros((num_samples, output_size))

    print('-------------------------------')
    print('Running subsample predictions')
    print('-------------------------------')


    for i in range(num_samples):

        # Find the indices of positions with 1 or -1
        positions = torch.nonzero(nn_input)
        
        if n == None:
            if i == 0:
                print('-------------------------------')
                print('Using all positions')
                print('-------------------------------')
            n = len(positions)

        if n > len(positions):
            if i == 0:
                print('-------------------------------')
                print('Using all positions')
                print('-------------------------------')

        random_ind = torch.randperm(len(positions), generator=torch.Generator().manual_seed(i))[:n]

        num_feature_input = int(random_ind.shape[0])

        selected_positions = positions[random_ind]

        # Create a new tensor with all zeros
        new_tensor = torch.zeros_like(nn_input)

        # Set the selected positions to their original values
        new_tensor[selected_positions] = nn_input[selected_positions]

        new_tensor = new_tensor.reshape(1, -1).to(device).to(torch.float32)
        
        outputs = None

        for k in range(4):
            if outputs == None:
                outputs = models[k](new_tensor)['y']
            else:
                outputs += models[k](new_tensor)['y']

        outputs = outputs / 4
        
        outputs = torch.nn.Softmax(dim=1)(outputs)

        result[i,:] = outputs.cpu().detach().numpy()

    df = pd.DataFrame(result, columns=class_names)

    df.to_csv(output_file, sep='\t', index=False)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i", 
        "--input_file", 
        help="path to input file"
    )

    parser.add_argument(
        "-m", 
        "--model", 
        help="specify path to model zip you want to use."
    )

    parser.add_argument(
        "-c", 
        "--num_CpGs", 
        help="specify the number of samples CpG sites (default is to use all available sites)."
    )

    parser.add_argument(
        "-o", 
        "--output_file", 
        help="input"
    )

    parser.add_argument(
        "-s", 
        "--num_samplings", 
        help="Specify the number of random samples of size num_CpGs. Default is 1 random sampling."
    )

    parser.add_argument(
        "-f", 
        "--file_type", 
        help="input file type 'bed' or 'csv'"
    )

    args = parser.parse_args()
    
    if args.num_samplings == None:
        num_samples = 1 
    else:
        num_samples = int(args.num_samplings)
    
    print('---------------------------------------------------------------------')
    print('Running intraoperative methylation based classification')
    print('---------------------------------------------------------------------')

    path_to_model_files = unpack_zip_file(args.model)

    predict(args.input_file, path_to_model_files, args.num_CpGs, args.output_file, num_samples, args.file_type)

def entrypoint():
    try:
        main()
    except Exception as e:
        import traceback
        print("\n❌ ERROR: An exception occurred in sturgeon-v2 CLI\n")
        traceback.print_exc()
        exit(1)

