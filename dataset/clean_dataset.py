from PIL import Image, ImageStat
import imagehash
import os
import sys
from tqdm import tqdm


class ImageRepo:
    def __init__(self, algo=imagehash.phash) -> None:
        self.algo = algo
        self.seen = set()

    def is_seen(self, img_path) -> bool:
        img = Image.open(img_path)
        hash = self.algo(img)

        if hash in self.seen:
            return True

        self.seen.add(hash)
        return False


def is_near_black_screen(img_path, threshold=20):
    img = Image.open(img_path).convert("RGB")
    stat = ImageStat.Stat(img)
    result = True
    for c in stat.mean:
        result &= c < threshold
    return result


def generate_list(scan_folder, result_file):
    paths = list()
    for dirpath, dirnames, filenames in os.walk(scan_folder):
        for filename in filenames:
            paths.append(os.path.join(dirpath, filename))
    paths.sort()

    repo = ImageRepo()
    near_black_list = list()
    duplicate_list = list()

    for path in tqdm(paths):
        if is_near_black_screen(path):
            near_black_list.append(path)
            continue

        if repo.is_seen(path):
            duplicate_list.append(path)
            continue

    with open(result_file, 'w') as f:
        for p in near_black_list:
            f.write(p)
            f.write('\n')

        f.write('\n')

        for p in duplicate_list:
            f.write(p)
            f.write('\n')


def execute(result_list):
    with open(result_list) as f:
        for line in f:
            path = line.strip()

            if path == '':
                continue

            try:
                os.remove(path)
            except:
                print("Failed to remove file: {}".format(path))


def usage(argv):
    print(
        """
{0} [command] [args]

command is one of: generate, execute

generate:
{0} generate scan_folder result_file

execute:
{0} execute result_file
""".format(argv[0])
    )
    sys.exit(-1)


if __name__ == '__main__':
    # Parse args
    argc = len(sys.argv)
    if argc < 2:
        usage(sys.argv)

    command = sys.argv[1]
    if command == 'generate':
        if argc < 4:
            usage(sys.argv)
        generate_list(sys.argv[2], sys.argv[3])
    elif command == 'execute':
        if argc < 3:
            usage(sys.argv)
        execute(sys.argv[2])
    else:
        usage(sys.argv)
