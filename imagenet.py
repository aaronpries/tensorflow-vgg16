# import gevent
import argparse
import grequests
import requests
from pprint import pprint
import os
import os.path
from tqdm import tqdm
from PIL import Image
from StringIO import StringIO
import random



def get_synsets(idfile="wnids.txt"):
  WNID_TO_SYNSET = "http://www.image-net.org/api/text/wordnet.synset.getwords?wnid={}"
  wanted = [l.strip() for l in open(idfile) if l.strip() is not '' and not l.startswith("#")]
  with open("fgo_synsets.txt", 'w') as fp:
    for wnid in wanted:
      l = ", ".join(requests.get(WNID_TO_SYNSET.format(wnid)).text.split("\n")[:-1])
      fp.write("{} {}\n".format(wnid, l))



def download_images(wnidfile, folder, n_images):
  def make_name(wnid, url):
    filename = url.encode("ascii", "ignore").replace("/","_")
    return os.path.join(folder, wnid, filename)

  URL = "http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}"
  wnids = [l.strip().split()[0] for l in open(wnidfile)]
  random.shuffle(wnids)
  session = requests.Session()
  for wnid in wnids:
    try:
      os.makedirs(os.path.join(folder, wnid))
    except os.error: pass
    res = requests.get(URL.format(wnid))
    urls = [_.strip() for _ in res.text.split("\n")]
    urls = [u for u in urls if u]
    print(len(urls))
    jobs = [grequests.get(url, session=session)
        for url in urls
        if not os.path.exists(make_name(wnid, url))
    ]
    n_already_have = (len(urls) - len(jobs))
    N = max(min(n_images, len(urls)) - n_already_have, 0)
    print("getting %s, (have %d, need %d) (%d/%d)" % (wnid, n_already_have, N, wnids.index(wnid)+1, len(wnids)))
    if N == 0: continue
    curr = 0
    pbar = tqdm(total=N)
    for res in grequests.imap(jobs, size=50):
      if curr >= N: break
      if "unavailable" in res.url:
        continue
      try:
        im = Image.open(StringIO(res.content))
        if im.width < 128 or im.height < 128: continue
        im.save(make_name(wnid, res.url))
        pbar.update()
        curr += 1
      except IOError: continue
      except Exception as e:
        # print("caught exception: %s" % e)
        continue


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--wnids', default="fgo_synsets.txt")
  parser.add_argument('--folder', default="imagenet")
  parser.add_argument('-n', default=1000, type=int)
  args = parser.parse_args()
  download_images(args.wnids, args.folder, args.n)
