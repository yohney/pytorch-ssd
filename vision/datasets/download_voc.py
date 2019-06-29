import urllib.request
import argparse
import os
import shutil
import tarfile

parser = argparse.ArgumentParser(description="VOC DS downloader.")
parser.add_argument('--destination', default="data/VOC",
                    help="Where to extract VOC DS.")
args = parser.parse_args()

dl_map = [
    {
        "extract_dir": "2007/test",
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
        "zipped_path": "VOCdevkit/VOC2007"
    },
    {
        "extract_dir": "2007/train",
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar",
        "zipped_path": "VOCdevkit/VOC2007"
    },
    {
        "extract_dir": "2012/train",
        "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
        "zipped_path": "VOCdevkit/VOC2012"
    }
]

for item in dl_map:
    print("Downloading: {}".format(item['extract_dir']))
    urllib.request.urlretrieve(item['url'], 'tmp.tar')
    print("Done. Extracting...")
    tar = tarfile.open("tmp.tar")
    tar.extractall()
    tar.close()

    dest = "{}/{}".format(args.destination, item['extract_dir'])
    print("Done. Moving to {}".format(dest))
    shutil.move(item['zipped_path'], dest)
    os.removedirs("VOCdevkit")
    print("Done")
    print("")

print("Cleanup...")
if os.path.isfile("tmp.tar"):
    os.remove('tmp.tar')