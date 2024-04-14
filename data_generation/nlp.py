import requests
import zipfile
import os
import errno
import nltk
from nltk.tokenize import sent_tokenize

#ALICE_URL = 'https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/1476/alice28-1476.txt'
#WIZARD_URL = 'https://ota.bodleian.ox.ac.uk/repository/xmlui/bitstream/handle/20.500.12024/1740/wizoz10-1740.txt'
ALICE_URL = 'https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/1476/alice28-1476.txt'
WIZARD_URL = 'https://llds.ling-phil.ox.ac.uk/llds/xmlui/bitstream/handle/20.500.14106/1740/wizoz10-1740.txt'
                
def download_text(url, localfolder='texts'):
    localfile = os.path.split(url)[-1]
    try:
        os.mkdir(f'{localfolder}')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        r = requests.get(url, allow_redirects=True)
        open(os.path.join(localfolder, localfile), 'wb').write(r.content)
    except Exception as e:
        print(f'Error downloading file: {str(e)}')
        
def sentence_tokenize(source, quote_char='\\', sep_char=',',
                      include_header=True, include_source=True, 
                      extensions=('txt'), **kwargs):
    nltk.download('punkt')
    # If source is a folder, goes through all files inside it
    # that match the desired extensions ('txt' by default)
    if os.path.isdir(source):
        filenames = [f for f in os.listdir(source)
                     if os.path.isfile(os.path.join(source, f)) and
                        os.path.splitext(f)[1][1:] in extensions]
    elif isinstance(source, str):
        filenames = [source]
    
    # If there is a configuration file, builds a dictionary with
    # the corresponding start and end lines of each text file
    config_file = os.path.join(source, 'lines.cfg')
    config = {}
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            rows = f.readlines()

        for r in rows[1:]:
            fname, start, end = r.strip().split(',')
            config.update({fname: (int(start), int(end))})
       
    new_fnames = []
    # For each file of text
    for fname in filenames:
        # If there's a start and end line for that file, use it
        try:
            start, end = config[fname]
        except KeyError:
            start = None
            end = None
            
        # Opens the file, slices the configures lines (if any)
        # cleans line breaks and uses the sentence tokenizer
        with open(os.path.join(source, fname), 'r') as f:
            contents = (''.join(f.readlines()[slice(start, end, None)])
                        .replace('\n', ' ').replace('\r', ''))
        corpus = sent_tokenize(contents, **kwargs)
        
        # Builds a CSV file containing tokenized sentences
        base = os.path.splitext(fname)[0]
        new_fname = f'{base}.sent.csv'
        new_fname = os.path.join(source, new_fname)
        with open(new_fname, 'w') as f:
            # Header of the file
            if include_header:
                if include_source:
                    f.write('sentence,source\n')
                else:
                    f.write('sentence\n')
            # Writes one line for each sentence
            for sentence in corpus:
                if include_source:
                    f.write(f'{quote_char}{sentence}{quote_char}{sep_char}{fname}\n')
                else:
                    f.write(f'{quote_char}{sentence}{quote_char}\n')
        new_fnames.append(new_fname)
        
    # Returns list of the newly generated CSV files
    return sorted(new_fnames)
