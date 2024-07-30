import json
import os
import argparse
import glob
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='generate splitted json file')

    parser.add_argument('--dir', default='data/omniobject3d_ocr/*/*')
    
    args = parser.parse_args()
    return args

def write_json_data(dict, output_dir):  # 写入json文件
    with open(output_dir, 'w') as r:
        json.dump(dict, r, indent=4)

def main():
    args = parse_args()
    work_dirs = glob.glob(os.path.join(args.dir, 'render'))
    for work_dir in tqdm.tqdm(work_dirs):
        with open(os.path.join(work_dir, 'transforms.json'), 'r') as f:
            anno = json.load(f)
        f.close()

        train, val, test = [{'camera_angle_x': anno['camera_angle_x'],
                            'frames': []} for _ in range(3)]
        for i, frame in enumerate(anno['frames']):
            frame['file_path'] = os.path.join('images', frame['file_path'])
            if i % 8 == 0:
                val['frames'].append(frame)
            elif i % 9 == 0:
                test['frames'].append(frame)
            else:
                train['frames'].append(frame)
        
        write_json_data(train, os.path.join(work_dir, 'transforms_train.json'))
        write_json_data(val, os.path.join(work_dir, 'transforms_val.json'))
        write_json_data(test, os.path.join(work_dir, 'transforms_test.json'))

if __name__ == '__main__':
    main()
