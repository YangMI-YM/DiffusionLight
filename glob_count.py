from glob import glob
import os
from tqdm.auto import tqdm
import argparse


def quick_counter(src_dir, des_dir):
    batches = os.listdir(src_dir)
    input_tol = 0
    processed_tol = 0
    for batch_name in batches:
        #if 'batch_1' not in batch_name:
        #    continue
        input_cnt = len(glob(os.path.join(src_dir, batch_name)+'/light_mask/*.*'))
        #output_cnt = len(glob(os.path.join(des_dir, batch_name)+'/control/*_ev-00.*'))
        output_cnt = len(glob(os.path.join(des_dir, batch_name)+'/light_mask/*.*'))
        input_tol += input_cnt
        processed_tol += output_cnt
        print(f"{batch_name}: Total: {input_cnt} | Processed: {output_cnt} | Remaining: {input_cnt-output_cnt}.")

    print(f"Progress: {processed_tol} / {input_tol}.")


def cross_check(src_dir, des_dir):
    batches = ['batch_2'] #os.listdir(des_dir)
    for batch_name in batches:
        filepath_list = glob(os.path.join(des_dir, batch_name)+'/control/*_ev-00.*')
        outlier_counter = 0
        for img_pth in tqdm(filepath_list):
            img_name = os.path.basename(img_pth).split('_ev-')[0]
            img_in_src = glob(os.path.join(src_dir, batch_name, img_name+'*.*'))
            if len(img_in_src) == 0:
                files_with_pattern = glob(os.path.join(des_dir, batch_name, 'control', img_name+'*.*'))
                outlier_counter +=1
                print(files_with_pattern) # safe check
                for matching_file in files_with_pattern:
                    try:
                        os.remove(matching_file) # remove from control
                        os.remove(matching_file.replace('control', 'raw')) # remove from raw
                        os.remove(matching_file.replace('control', 'square')) # remove from square
                    except FileNotFoundError as e:
                        print(e)
        print(f"enfin: {outlier_counter}.")


if __name__ in '__main__':

    #beauty_dir = '/home/yangmi/s3data-2/beauty-lvm/v2/cropped/1024/'
    #output_dir = '/home/yangmi/s3data-2/beauty-lvm/v2/light/1024/' # modality: {'HED', 'color', 'SimpleHED', 'MaskHED', 'light'}

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--resolution", type=int, required=True)
    args = parser.parse_args()
    task = args.task
    img_dim = args.resolution
    beauty_dir = f'/home/yangmi/s3data-2/beauty-lvm/v2/light/{img_dim}/'
    output_dir = f'/home/yangmi/volume/light/{img_dim}/'

    if task == 'count':
        quick_counter(beauty_dir, output_dir)
    elif task == 'check':
        cross_check(beauty_dir, output_dir)
    else:
        raise NotImplementedError