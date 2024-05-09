import yaml
import os
import cv2
import sys

def read_config(PATH_CONFIG = "../config.yml"):
    with open(PATH_CONFIG, "r") as file:
        config = yaml.safe_load(file)
    return config

def decrypt_labels(labels):
    if labels[0] == 1:
        return 0 #fire
    elif labels[1] == 1:
        return 1 #smoke
    else:
        return 2 #non fire


def output_to_file(filename):
    orig_stdout = sys.stdout
    f = open(filename, "w")
    sys.stdout = f
    return orig_stdout, f

def output_as_before(std_orig, file):
    sys.stdout = std_orig
    file.close()


def run_check_images(sdir):
    ext_list = ['jpg', 'jpeg', 'png']
    color_mode = 'rgb'
    good_count, bad_count, bad_files = check_images(sdir, ext_list, color_mode)
    print('number of good files is: ', good_count, '\nnumber of bad files is: ', bad_count)
    if bad_count > 0:
        for f in bad_files: \
            print(f)

# code taken from https://stackoverflow.com/questions/69252271/tf-keras-preprocessing-image-dataset-from-directory-is-not-reading-all-image-fro
def check_images(sdir, ext_list, color_mode):
    def inc_bad(bad_count, bad_list, fpath):
        bad_count += 1
        bad_list.append(fpath)
        return bad_count, bad_list

    def check_color_mode(channels, color_mode, bad_count, bad_list, fpath):
        result = True
        if color_mode == ('rgb' and channels != 3) or (color_mode == 'rgba' and channels != 4) or (
                color_mode == 'grayscale' and channels != 1):
            bad_count, bad_list = inc_bad(bad_count, bad_list, fpath)
            result = False
        return result, bad_count, bad_list

    good_count = 0
    bad_count = 0
    bad_list = []
    classlist = os.listdir(sdir)
    print('The classes found are \n', classlist, '\n')
    for klass in classlist:
        print('Processing class ', klass, '\n')
        classpath = os.path.join(sdir, klass)
        if os.path.isdir(classpath):
            flist = os.listdir(classpath)
            for f in flist:
                fpath = os.path.join(classpath, f)
                if os.path.isfile(fpath):
                    index = f.rfind('.')
                    ext = f[index + 1:]
                    if ext.lower() in ext_list:
                        try:
                            img = cv2.imread(fpath)
                            channels = img.shape[2]
                            result, bad_count, bad_list = check_color_mode(channels, color_mode, bad_count, bad_list,
                                                                           fpath)
                            if result:
                                good_count += 1  # check_color_mode found no error
                        except:
                            bad_count, bad_list = inc_bad(bad_count, bad_list, fpath)

                    else:
                        bad_count, bad_list = inc_bad(bad_count, bad_list, fpath)
    return good_count, bad_count, bad_list