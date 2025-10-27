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

> **Recommendation:** For optimal performance, set -c between 10,000 and 20,000 CpG sites. Using too many CpG sites may cause the model to become overconfident and increase the likelihood of misclassification.
> >

## Preparing data into the right format: bed or csv
Tucan accepts input in either **BED** or **CSV** format, containing the methylation calls.  
An example BED file snippet is available here: [data/bedExample.bed](data/bedExample.bed).

Methylation calls can be extracted from a Nanopore BAM file using **[modkit](https://github.com/nanoporetech/modkit)**.  
Detailed instructions for this process are available in the [Sturgeon repository](https://github.com/UMCUGenetics/sturgeon).

> **Note:** The Sturgeon implementation determines methylation status at CpG sites based on the **Illumina methylation array coordinates**, including a **±25 bp window** around each CpG site.  
> This means methylation calls are aggregated not only at the exact array position but also within this surrounding margin.

## Table: Abbreviation - Tumor Type
See the full table: [docs/tumor_abbreviation.md](docs/tumor_abbreviation.md)
