import urllib.request
import os
import urllib
import shutil

''''''
for i in range(200000):
    urllib.request.urlretrieve('http://localhost:8080','./data/train/curr.jpg')
    shutil.move('./data/train/curr.jpg','./data/train/{}.jpg'.format(urllib.request.urlopen('http://localhost:8080/index.php/home/index/code').read().decode('UTF-8')))
    print('train-{}'.format(i))

for i in range(50000):
    urllib.request.urlretrieve('http://localhost:8080','./data/test/curr.jpg')
    shutil.move('./data/test/curr.jpg','./data/test/{}.jpg'.format(urllib.request.urlopen('http://localhost:8080/index.php/home/index/code').read().decode('UTF-8')))
    print('train-{}'.format(i))




