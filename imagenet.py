# import gevent
import grequests
import requests
from pprint import pprint
import os
import os.path
from tqdm import tqdm


def get_synsets(idfile="wnids.txt"):
  WNID_TO_SYNSET = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={}"
  wanted = [l.strip() for l in open(idfile) if l.strip() is not '' and not l.startswith("#")]
  with open("fgo_synsets.txt", 'w') as fp:
    for wnid in wanted:
      l = ", ".join(urllib.request.urlopen(WNID_TO_SYNSET.format(wnid)).read().decode("utf-8").split("\n")[:-1])
      fp.write("{} {}\n".format(wnid, l))


def download_images(wnidfile="fgo_synsets.txt", folder="imagenet"):
  URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}"
  wnids = [l.strip().split()[0] for l in open(wnidfile)]
  for wnid in wnids:
    print("getting %s (%d/%d)" % (wnid, wnids.index(wnid)+1, len(wnids)))
    try:
      os.makedirs(os.path.join(folder, wnid))
    except: pass
    urls = [_.strip() for _ in requests.get(URL.format(wnid)).text.split("\r")]
    jobs = [grequests.get(url) for url in urls]
    pbar = tqdm(total=len(jobs))
    for res in grequests.imap(jobs, size=50):
      pbar.update()
      filename = res.url.replace("/","_")
      with open(os.path.join(folder, wnid, filename), 'wb') as fp:
        fp.write(res.content)
    # [job.link(store_image(wnid)) for job in jobs]

if __name__ == '__main__':
  download_images()
