# tucan
This repository presents a pip installable software package to easily run tucan in a clinical setting.

## Installation
* Install using `pip install git+ssh://git@github.com/UMCUGenetics/tucan.git`
* Download the tucan [zip file](https://filesender.surf.nl/?s=download&token=540a148b-a695-4ad7-a303-2f320dddf484)
* Download the model from Hugging Face at runtime
  pip install -U huggingface_hub
  export HUGGINGFACE_HUB_TOKEN=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
  python -c "from tucan.download_model import get_model; print(get_model())"

## Usage
<pre>
Usage: sturgeon-v2 [-h] [-i INPUT_FILE] [-m MODEL] [-c NUM_CPGS] [-o OUTPUT_FILE] [-s NUM_SAMPLINGS]
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
                        input
  -s NUM_SAMPLINGS, --num_samplings NUM_SAMPLINGS
                        Specify the number of random samples of size num_CpGs. Default is 1 random
                        sampling.
  -f FILE_TYPE, --file_type FILE_TYPE
                        input file type 'bed' or 'csv'
</pre>
