pip install -U duckduckgo_search

from duckduckgo_search import ddg_images
from fastcore.all import *
from fastdownload import download_url
from fastai.vision.all import *
from time import sleep
import pickle

def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image')

urls = search_images('hippo photos', max_images=1)
urls[0]

dest = 'hippo.jpg'
download_url(urls[0], dest, show_progress=True)

im = Image.open(dest)
im.to_thumb(256,256)

download_url(search_images('forest photos', max_images=1)[0], 'forest.jpg', show_progress=False)
Image.open('forest.jpg').to_thumb(256,256)

searches = 'forest','hippo'
path = Path('hippo_or_not')

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)

failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)

dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # The inputs to the model are images, and the outputs are categories (in this case, "hippo" or "forest")
    get_items=get_image_files, # To find all the inputs to the model, run the get_image_files function (which returns a list of all image files in a path)
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # Split the data into training and validation sets randomly, using 20% of the data for the validation set
    get_y=parent_label, # The labels (y values) is the name of the parent of each file (i.e. the name of the folder they're in, which will be bird or forest)
    item_tfms=[Resize(192, method='squish')] # Before training, resize each image to 192x192 pixels by "squishing" it (as opposed to cropping it).
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)

learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)

is_hippo,_,probs = learn.predict(PILImage.create('gloria.jpg'))
print(f"This is a: {is_hippo}.")
print(f"Probability it's a forest: {probs[0]:.4f}")