import os
import sys
import errno
import requests
import subprocess
import shutil
from IPython.display import HTML, display
from tensorboard import manager

def tensorboard_cleanup():
    info_dir = manager._get_info_dir()
    shutil.rmtree(info_dir)

FOLDERS = {
    0: ['plots'],
    1: ['plots'],
    2: ['plots', 'data_generation', 'data_preparation', 'model_configuration', 'model_training'],
    21: ['plots', 'data_generation', 'data_preparation', 'model_configuration', 'stepbystep'],
    3: ['plots', 'stepbystep'],
    4: ['plots', 'stepbystep', 'data_generation'],
    5: ['plots', 'stepbystep', 'data_generation', ''],
    6: ['plots', 'stepbystep', 'stepbystep', 'data_generation', 'data_generation', 'data_preparation'],
    7: ['plots', 'stepbystep', 'data_generation'],
    71: ['plots', 'stepbystep', 'data_generation'],
    8: ['plots', 'plots', 'stepbystep', 'data_generation'],
    9: ['plots', 'plots', 'plots', 'stepbystep', 'data_generation'],
    10: ['plots', 'plots', 'plots', 'plots', 'stepbystep', 'data_generation', 'data_generation', '', ''],
    11: ['plots', 'stepbystep', 'data_generation', ''],
}
FILENAMES = {
    0: ['chapter0.py'],
    1: ['chapter1.py'],
    2: ['chapter2.py', 'simple_linear_regression.py', 'v0.py', 'v0.py', 'v0.py'],
    21: ['chapter2_1.py', 'simple_linear_regression.py', 'v2.py', '', 'v0.py'],
    3: ['chapter3.py', 'v0.py'],
    4: ['chapter4.py', 'v0.py', 'image_classification.py'],
    5: ['chapter5.py', 'v1.py', 'image_classification.py', 'helpers.py'],
    6: ['chapter6.py', 'v2.py', 'v3.py', 'rps.py', 'simple_linear_regression.py', 'v2.py'],
    7: ['chapter7.py', 'v3.py', 'rps.py'],
    71: ['chapterextra.py', 'v3.py', 'ball.py'],
    8: ['chapter8.py', 'replay.py', 'v4.py', 'square_sequences.py'],
    9: ['chapter8.py', 'chapter9.py', 'replay.py', 'v4.py', 'square_sequences.py'],
    10: ['chapter8.py', 'chapter9.py', 'chapter10.py', 'replay.py', 'v4.py', 'square_sequences.py', 'image_classification.py', 'helpers.py', 'seq2seq.py'],
    11: ['chapter11.py', 'v4.py', 'nlp.py', 'seq2seq.py'],
}

try:
    host = os.environ['BINDER_SERVICE_HOST']
    IS_BINDER = True
except KeyError:
    IS_BINDER = False
    
try:
    import google.colab
    IS_COLAB = True
except ModuleNotFoundError:
    IS_COLAB = False

IS_LOCAL = (not IS_BINDER) and (not IS_COLAB)

def download_to_colab(chapter, branch='master'):    
    base_url = 'https://raw.githubusercontent.com/dvgodoy/PyTorchStepByStep/{}/'.format(branch)

    folders = FOLDERS[chapter]
    filenames = FILENAMES[chapter]
    for folder, filename in zip(folders, filenames):
        if len(folder):
            try:
                os.mkdir(folder)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

        if len(filename):
            path = os.path.join(folder, filename)
            url = '{}{}'.format(base_url, path)
            r = requests.get(url, allow_redirects=True)
            open(path, 'wb').write(r.content)

    try:
        os.mkdir('runs')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

TB_LINK = ''
if IS_BINDER:
    TB_LINK = HTML('''
    <a href="" target="_blank" id="tb">Click here to open TensorBoard</a>
    <script>
        var address=document.location.href;
        a = document.getElementById('tb');
        a.href = address.substr(0, address.lastIndexOf("/")-9).concat("proxy/6006/");
    </script>
    ''')
    
def config_chapter0(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(0, branch)
        print('Finished!')
    
def config_chapter1(branch='master'):
    if IS_COLAB:
        print('Installing torchviz...')
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'torchviz'])
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(1, branch)
        print('Creating folders...')
        folders = ['data_preparation', 'model_configuration', 'model_training']

        for folder in folders:
            try:
                os.mkdir(folder)
            except OSError as e:
                e.errno
                if e.errno != errno.EEXIST:
                    raise
        print('Finished!')
        
def config_chapter2(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(2, branch)
        print('Finished!')

def config_chapter2_1(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(21, branch)
        print('Finished!')

def config_chapter3(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(3, branch)
        print('Finished!')

def config_chapter4(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(4, branch)
        print('Finished!')

def config_chapter5(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(5, branch)
        print('Finished!')

def config_chapter6(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(6, branch)
        print('Finished!')
        
def config_chapter7(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(7, branch)
        print('Finished!')
        
def config_chapterextra(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(71, branch)
        print('Finished!')
        
def config_chapter8(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(8, branch)
        print('Finished!')
        
def config_chapter9(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(9, branch)
        print('Finished!')

def config_chapter10(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(10, branch)
        print('Finished!')

def config_chapter11(branch='master'):
    if IS_COLAB:
        print('Downloading files from GitHub repo to Colab...')
        download_to_colab(11, branch)
        print('Finished!')
