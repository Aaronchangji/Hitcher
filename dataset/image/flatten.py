import os
import sys
import shutil

count = 0

def get_jpgs(path, save_path):
    global count
    files = os.listdir(path)
    for file in files:
        sub_path = os.path.join(path, file)
        if os.path.isdir(sub_path):
            get_jpgs(sub_path, save_path)
        elif "jpg" in file:
            new_file = "{0}.jpg".format(count)
            count += 1
            new_file = os.path.join(save_path, new_file)
            # shutil.move(sub_path, new_file)
            print(new_file)

if __name__ == '__main__':
    path = sys.argv[1]
    save_path = sys.argv[2]
    get_jpgs(path, save_path)
    print("total image number: {0}".format(count))
    