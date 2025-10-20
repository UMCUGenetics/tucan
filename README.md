# Tucan
This repository presents a pip installable software package to easily run tucan in a clinical setting.

## Installation 
* `git clone https://github.com/UMCUGenetics/tucan.git`
* `cd tucan`
* Install the project in 'editable' mode `pip install -e .` 
* Download the model from Hugging Face 
<pre>
pip install -U huggingface_hub
python -c "from src.tucan.download_model import get_model; print(get_model())"
</pre>
* changle dir `cd models`
* zip file
`zip -r model.zip model`  


## Usage
<pre>
Usage: tucan [-h] [-i INPUT_FILE] [-m MODEL] [-c NUM_CPGS] [-o OUTPUT_FILE] [-s NUM_SAMPLINGS]
                   [-f FILE_TYPE]

Options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        path to input file
  -m MODEL, --model MODEL
                        specify path to model zip you want to use.
  -c NUM_CPGS, --num_CpGs NUM_CPGS
                        specify the number of samples CpG sites (default is to use all available sites).
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        path to output file
  -s NUM_SAMPLINGS, --num_samplings NUM_SAMPLINGS
                        Specify the number of random samples of size num_CpGs. Default is 1 random
                        sampling.
  -f FILE_TYPE, --file_type FILE_TYPE
                        input file type 'bed' or 'csv'
</pre>


## Table: Abbreviation - Tumor Type
See the full table: [docs/tumor_abbreviation.md](docs/tumor_abbreviations.md)

