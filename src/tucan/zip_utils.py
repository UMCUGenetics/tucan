import os
import zipfile

def unpack_zip_file(zip_path):
    # Get the absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # The 'include' directory is two levels up from this script
    include_dir = os.path.abspath(os.path.join(script_dir, '..', '..', 'include'))

    # Ensure the include directory exists
    os.makedirs(include_dir, exist_ok=True)

    # Check if the zip file has already been extracted (by checking a folder with the same name)
    zip_base = os.path.splitext(os.path.basename(zip_path))[0]
    potential_extract_dir = os.path.join(include_dir, zip_base)
    if os.path.exists(potential_extract_dir) and os.path.isdir(potential_extract_dir):

        print('-------------------------------')
        print('already unzipped')
        print('-------------------------------')

        return potential_extract_dir

    
    print('-------------------------------')
    print('unzipping')
    print('-------------------------------')

    # Extract directly into the include directory
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(include_dir)

    return potential_extract_dir
