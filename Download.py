import urllib.request
import time
import sys
import time
import urllib
import tarfile
import gzip


def reporthook(count, block_size, total_size):
    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    if duration == 0.0:
        duration += 0.00001
    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(int(count*block_size*100/total_size),100)
    #percent = int(count * block_size * 100 / total_size)
    sys.stdout.write("\r...%d%%, %d MB, %d KB/s, %d seconds passed" %
                     (percent, progress_size / (1024 * 1024), speed, duration))
    sys.stdout.flush()



def download(url, path="./"):
    # Download the file from `url` and save it locally under `file_name`:
    split = urllib.parse.urlsplit(url)
    path = path + split.path.split("/")[-1]
    file_name, headers = urllib.request.urlretrieve(url, path, reporthook=reporthook)
    print("\nDownload complete!")
    return path

def tar_gz(path):
    # open the tarfile and use the 'r:gz' parameter
    # the 'r:gz' mode enables gzip compression reading
    tfile = tarfile.open(path, 'r:gz')

    # 99.9% of the time you just want to extract all
    # the contents of the archive.
    tfile.extractall('.')

    # Maybe this isn't so amazing for you types out
    # there using *nix, os x, or (anything other than
    # windows that comes with tar and gunzip scripts).
    # However, if you find yourself on windows and
    # need to extract a tar.gz you're in for quite the
    # freeware/spyware/spamware gauntlet.

    # Python has everything you need built in!
    # Hooray for python!

def gz(path):
    out_file = path.replace(".gz","")
    with gzip.open(path, 'rb') as f:
        file_content = f.read()
        with open(out_file , "wb")as file:
            file.write(file_content)
    return out_file