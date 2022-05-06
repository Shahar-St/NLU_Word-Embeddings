import math
import os.path
import string
import xml.etree.ElementTree as ET
from collections import Counter
from sys import argv

import numpy as np
from gensim.models import KeyedVectors


class Token:

    def __init__(self, value):
        self.value = value


class Sentence:
    def __init__(self, tokens_array):
        self.tokens = tokens_array


class Corpus:
    def __init__(self):
        self.sentences = []
        self.chunks = []

    def add_xml_file_to_corpus(self, file_name: str):
        tree = ET.parse(file_name)
        # iterate over all sentences in the file and extract the tokens
        for sentence in tree.iter(tag='s'):
            tokens = np.array([Token(value=word.text.strip()) for word in sentence if
                               word.tag in ('w', 'c') and isinstance(word.text, str)])

            new_sentence = Sentence(tokens)
            self.sentences.append(new_sentence)

    def get_tokens(self):
        # get a list of all tokens in their lower case form
        tokens_list = []
        for sen in self.sentences:
            tokens_list.extend([tok.value for tok in sen.tokens])
        return tokens_list


class NGramModel:
    def __init__(self, max_n, corpus, linear_interpolation_params: tuple):
        """
        The NGramModel object holds several models' calculation and is defined by the max_n param
        i.e: If max_n = 3, the obj will be able to perform as unigarm, Bigram and Trigram models.
        All (public) methods accept the n that determines on which model the method will work
        """
        self.linear_interpolation_params = linear_interpolation_params
        self.corpus = corpus
        self.max_n = max_n
        self.voc_sizes = []

        # Create unigram model
        tokens = self.corpus.get_tokens()
        self.voc_sizes.append(len(set(tokens)))
        self.num_of_words = len(tokens)

        n_words_combinations = [tokens]

        # create Ngram model for k = 2 to max_n
        for k in range(2, max_n + 1):
            k_combinations = []
            for i in range(len(tokens) - k + 1):
                k_combinations.append(' '.join(tokens[i:i + k]))
            n_words_combinations.append(k_combinations)
            self.voc_sizes.append(len(set(k_combinations)))

        counters = [dict(Counter(combination)) for combination in n_words_combinations]
        self.n_tokens_counters = counters


def main():
    # init_key_vectors()

    # kv_file = argv[1]
    kv_file_name = '300d.kv'
    kv_file = os.path.join(os.getcwd(), 'key_vectors', kv_file_name)
    # xml_dir = argv[2]  # directory containing xml files from the BNC corpus (not a zip file)
    xml_dir = os.path.join(os.getcwd(), 'XML_files')
    # lyrics_file = argv[3]
    lyrics_file = os.path.join(os.getcwd(), 'leave_the_door_open.txt')
    # tweets_file = argv[4]
    tweets_file = os.path.join(os.getcwd(), 'tweets.txt')
    # output_file = argv[5]
    output_file = os.path.join(os.getcwd(), 'output.txt')

    model: KeyedVectors = KeyedVectors.load(kv_file, mmap='r')

    # Task A
    task_a_str = warm_up_task(model)

    # Task B
    task_b_str = grammy(lyrics_file, xml_dir, model)

    # Task C
    tweets(tweets_file)

    print_output(output_file, task_a_str, task_b_str)
    print(f'Program ended.')


def init_key_vectors():
    print('Initializing key vectors')
    from gensim.scripts.glove2word2vec import glove2word2vec

    glove_input_dir = 'GloVe_text_files'
    word2vec_output_dir = 'word2vec_files'
    key_vectors_dir = 'key_vectors'

    for i, file in enumerate(os.listdir(glove_input_dir)):
        file_num = file.split('.')[2]
        downloaded_text_filename = os.path.join(os.getcwd(), glove_input_dir, file)
        full_path_vector_filename = os.path.join(os.getcwd(), word2vec_output_dir, file_num)
        glove2word2vec(downloaded_text_filename, full_path_vector_filename)
        pre_trained_model = KeyedVectors.load_word2vec_format(full_path_vector_filename, binary=False)
        pre_trained_model.save(os.path.join(os.getcwd(), key_vectors_dir, f'{file_num}.kv'))

    print('Created Key vectors')


def warm_up_task(pre_trained_model):
    print('Warm up task - In Progress...')
    # Task 1 - Pairs similarity
    pairs_list = [
        ('president', 'state'),
        ('apple', 'pear'),
        ('hate', 'love'),
        ('university', 'school'),
        ('house', 'home'),
        ('planet', 'star'),
        ('near', 'next'),
        ('swim', 'dive'),
        ('hard', 'easy'),
        ('sunrise', 'dawn'),
    ]
    output_str = 'Word Pairs and Distances:\n'
    for i, pair in enumerate(pairs_list):
        dist = pre_trained_model.similarity(pair[0], pair[1])
        output_str += f'{i}. {pair[0]} - {pair[1]} : {dist}\n'

    # Task 2 - Pairs analogy
    analogies_list = [
        (('funny', 'humorous'), ('hardworking', 'diligent')),
        (('man', 'woman'), ('boy', 'girl')),
        (('itch', 'scratch'), ('virus', 'cold')),
        (('read', 'learn'), ('try', 'improve')),
        (('broom', 'sweep'), ('paintbrush', 'paint')),
    ]

    analogies_str = 'Analogies:\n'
    most_similar_str = 'Most Similar:\n'
    distances_str = 'Distances:\n'
    for i, (pair1, pair2) in enumerate(analogies_list):
        analogies_str += f'{i}. {pair1[0]} : {pair1[1]} , {pair2[0]} : {pair2[1]}\n'
        returned_word = pre_trained_model.most_similar(positive=[pair2[0], pair1[1]], negative=[pair1[0]])[0][0]
        most_similar_str += f'{i}. {pair2[0]} + {pair1[1]} - {pair1[0]} = {returned_word}\n'
        expected_word = pair2[1]
        dist = pre_trained_model.similarity(returned_word, expected_word)
        distances_str += f'{i}. {expected_word} - {returned_word} : {dist}\n'

    output_str += f'\n{analogies_str}\n{most_similar_str}\n{distances_str}\n'
    print('Warm up task - Done.')
    return output_str


def grammy(lyrics_file, xml_dir, model: KeyedVectors):
    corpus = Corpus()
    xml_files_names = os.listdir(xml_dir)
    for file in xml_files_names:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    max_n = 3
    linear_interpolation_params = (0.2, 0.35, 0.45)
    n_gram_model = NGramModel(max_n, corpus, linear_interpolation_params)

    words_to_change = [
        'baby',
        'doing',
        'at',
        'plans',
        'say',
        'wine',
        'look',
        'house',
        'newborn',
        'dancing',
        'wing',
        'mansion',
        'playing',
        'heart',
        'arms',
        'door',
        'door',
        'door',
        'door',
        'feel',  # todo line 39

    ]
    print('Grammy Task - In Progress...')
    output_str = '=== New Hit ===\n'
    with open(lyrics_file) as lyrics_txt:
        for line, word_to_change in zip(lyrics_txt, words_to_change):
            line = line.replace('\n', '')
            tokens = tokenize_sentence(line)
            tokens_len = len(tokens)
            optional_replacements = model.most_similar(word_to_change)

            original_word_indices = [i for i in range(tokens_len) if tokens[i] == word_to_change]
            for ind in original_word_indices:
                # todo ask if that's what she meant
                start_ind_to_search = max(ind - 1, 0)
                end_ind_to_search = min(ind + 1, tokens_len - 1)
                relevant_tokens = tokens[start_ind_to_search:end_ind_to_search + 1]
                n_gram_num = len(relevant_tokens) - 1
                max_count = 0
                best_match = optional_replacements[0][0]
                for replacement, _ in optional_replacements:
                    search_text = ' '.join(relevant_tokens).replace(word_to_change, replacement)
                    count = n_gram_model.n_tokens_counters[n_gram_num].get(search_text, 0)
                    if count > max_count:
                        max_count = count
                        best_match = replacement
                if max_count == 0:  # no match found in trigram
                    for replacement, _ in optional_replacements:
                        # todo what happens if that's the last/first one too
                        first_part = tokens[start_ind_to_search: ind + 1]
                        first_part_len = len(first_part)
                        first_part_to_search = ' '.join(first_part).replace(word_to_change, replacement)
                        sec_part = tokens[ind: end_ind_to_search + 1]
                        sec_part_len = len(sec_part)
                        sec_part_to_search = ' '.join(sec_part).replace(word_to_change, replacement)
                        count = n_gram_model.n_tokens_counters[first_part_len - 1].get(first_part_to_search, 0) + \
                                n_gram_model.n_tokens_counters[sec_part_len - 1].get(sec_part_to_search, 0)
                        if count > max_count:
                            max_count = count
                            best_match = replacement

            new_line = line.replace(word_to_change, best_match)
            output_str += new_line + '\n'

    print('Grammy Task - Done')
    return output_str


def tokenize_sentence(sentence):
    for sign in string.punctuation:
        sentence = sentence.replace(sign, ' ' + sign + ' ')

    tokens = list(filter(lambda token: token != '', sentence.split(' ')))
    return tokens


def tweets(tweets_file):
    pass


def print_output(output_file, task_a_str, task_b_str):
    output_str = task_a_str + task_b_str
    print(f'Writing output to {output_file}')
    with open(output_file, 'w', encoding='utf8') as output_file:
        output_file.write(output_str)


if __name__ == "__main__":
    main()
