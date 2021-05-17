# Using ACE for Concept Generation

### 0. Requirements

You need to have ImageNet data somewhere. Change the `IMAGE_NET_PATH` constant to ImageNet directory.

### 1. Environment

We recommend using Anaconda with `environment.yml` file provided:

```
conda env create -f environment.yml
conda activate ace
``` 

We use:

```
    - python=3.6.12
    - pytorch=1.7.0=py3.6_cuda10.1.243_cudnn7.6.3_0
    - tensorflow-gpu==1.15.0
```

Module for automatic visual words generation `ACE` comes from `https://github.com/amiratag/ACE`. It is licensed under the MIT License. We introduce minor changes to this module, not related to the algorithm.

### 2. Data generation

#### 2.1 Visual words (concepts) generation

For desired target class from ImageNet, let say `zebra` run:
```
python create_data.py zebra
```
This will create `data` folder in your current directory and populate it needed data. 

For the list of ImageNet classes check the `imagenet_dirs.txt` file.

If everything worked you should be able to run this code. `num_random_exp` is set to `1` for a quick test, change it later I guess (`create_data.py` creates dirs for up to 10 `num_random_exp`, change it if necessary there).
We will move to `ACE` directory to use its default arguments which would need to be changed otherwise
```
cd ACE
python ace_run.py --target_class zebra --source_dir ../data --working_dir ../results --num_random_exp 1
```

There will be *lots* of warnings, deprecation and other and not really any logging (we are using their up to date code). 
You should expect the script to run for few minutes, afterwards there should be a `results` directory in this dir.

After running this script concepts will be saved at `{target_class}_dict.pkl`.

#### 2.2 Generate data for Word Content probing task

For WC probing task we need labels (list of concepts in the image) and embeddings (self-supervised representations). We also need generate superpixels' activation in the embedding space of supervised model in order to calculate labels. These are generated as follows:

```
python generate_data_for_probings.py --generate activations --classes CLASSNAME_0 CLASSNAME_1 ...
python generate_data_for_probings.py --generate labels --classes CLASSNAME_0 CLASSNAME_1 ...
python generate_data_for_probings.py --generate embeddings --classes CLASSNAME_0 CLASSNAME_1 ...
```
We take only top 100 most relevant visual words for the probing tasks (WC and SL). They are calculated with `generate_top_tcav_concepts.ipynb`.

#### 2.3 Generate data for Sentence Length probing task.

For SL probing task we need labels (number of unique concepts in the image) and embeddings (self-supervised representations). Since the number of unique concepts in the image is derived from the list of all concepts, we don't need to generate anything extra if we generated data for the WC probing task.

#### 2.4 Generate data for Character Bin probing task.

For CB probing task we need to generate patches (images of superpixels) and their embeddings. These are generated as follows:


```
python generate_data_for_probings.py --generate patches --classes CLASSNAME_0 CLASSNAME_1 ...
python generate_data_for_probings.py --generate patch_embeddings --classes CLASSNAME_0 CLASSNAME_1 ...
```

#### 2.5 Generate data for Semantic Odd Man Out probing task.

For SOMO probing task, we need to generate images with randomly blended superpixels from other images. These are generated as follows:

```
python generate_data_for_probings.py
```

### 3. Running probing tasks.

After all embeddings and labels have been generated, you can evaluate the representations using different probing tasks.

#### 3.1 Running Word Content probing task.

Run `word_content.ipynb` interactive notebook.

#### 3.2 Running Sentence Length probing task.

Run `python run_sentence_length.py`.

#### 3.3 Running Character Bin probing task.

Run `python run_character_bin.py`.

#### 3.4 Running Semantic Odd Man Out probing task.

Run `python run_somo.py`.
