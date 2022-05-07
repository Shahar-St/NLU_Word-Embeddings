import os.path
import re
import xml.etree.ElementTree as ET
from collections import Counter
from random import randrange
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import KeyedVectors
from sklearn.decomposition import PCA


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
    def __init__(self, max_n, corpus):
        """
        The NGramModel object holds several models' calculation and is defined by the max_n param
        i.e: If max_n = 3, the obj will be able to perform as Unigarm, Bigram and Trigram models.
        """
        self.corpus = corpus
        self.max_n = max_n
        self.voc_sizes = []

        # Create Unigram model
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


class Tweet:
    def __init__(self, tweet_text, category, index):
        self.index = index
        self.category = category
        self.tweet_text = tweet_text
        self.tokens = tokenize_sentence(tweet_text)

    def get_new_vector(self, weight_func, model: KeyedVectors):
        # calculate the new vector given a weight function
        w = np.array([np.full(model.vector_size, weight_func(tok)) for tok in self.tokens])
        v = np.array(
            [model[tok.lower()] if tok.lower() in model else np.ones(model.vector_size) for tok in self.tokens]
        )
        k = len(self.tokens)
        new_vec = np.zeros(model.vector_size)
        for i in range(k):
            new_vec += np.multiply(w[i], v[i])
        new_vec = new_vec / k
        return new_vec


def main():
    # init_key_vectors()

    kv_file = argv[1]
    xml_dir = argv[2]  # directory containing xml files from the BNC corpus (not a zip file)
    lyrics_file = argv[3]
    tweets_file = argv[4]
    output_file = argv[5]

    model: KeyedVectors = KeyedVectors.load(kv_file, mmap='r')

    task_a_str = warm_up_task(model)

    task_b_str = grammy(lyrics_file, xml_dir, model)

    tweets(tweets_file, model)

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
    print('Grammy Task - In Progress...')
    # init corpus and n-gram model
    corpus = Corpus()
    xml_files_names = os.listdir(xml_dir)
    for file in xml_files_names:
        corpus.add_xml_file_to_corpus(os.path.join(xml_dir, file))
    max_n = 3
    n_gram_model = NGramModel(max_n, corpus)

    # words_to_change[i] = the word to replace in line i. This will replace all occurrences of the word in line
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
        'feel',
        'want',
        'coming',
        'sweet',
        'bite',
        'smoke',
        'hungry',
        'baby',
        'love',
        'kissing',
        'bathtub',
        'bubbling',
        'playing',
        'heart',
        'arms',
        'door',
        'door',
        'door',
        'door',
        'feel',
        'want',
        'girl',
        'baby',
        'baby',
        'girl',
        'ah',
        'door',
        'door',
        'door',
        'feel',
        'want',
        'tell',
        'tell',
        'tell',
        'woo',
        'woo',
        'la',
        'tell',
        'waiting',
        'adore',
        'waiting',
        'tell',
        'girl',
        'adore',
        'la'
    ]
    output_str = '=== New Hit ===\n'
    with open(lyrics_file) as lyrics_txt:
        for line, word_to_change in zip(lyrics_txt, words_to_change):
            line = line.replace('\n', '')
            tokens = tokenize_sentence(line)
            tokens_len = len(tokens)
            optional_replacements = model.most_similar(word_to_change)

            # get all indices to replace
            original_word_indices = [i for i in range(tokens_len) if tokens[i].lower() == word_to_change.lower()]
            for ind in original_word_indices:
                # Trigram
                if ind != 0 and ind != tokens_len - 1:
                    start_ind_to_search = ind - 1
                    end_ind_to_search = ind + 1
                elif ind == 0:  # first word
                    start_ind_to_search = ind
                    end_ind_to_search = ind + 2
                else:  # last word
                    start_ind_to_search = ind - 2
                    end_ind_to_search = ind
                relevant_tokens = tokens[start_ind_to_search:end_ind_to_search + 1]
                max_count = 0
                best_match = optional_replacements[0][0]
                for replacement, _ in optional_replacements:
                    search_text = ' '.join(relevant_tokens).replace(word_to_change, replacement)
                    count = n_gram_model.n_tokens_counters[2].get(search_text, 0)
                    if count > max_count:
                        max_count = count
                        best_match = replacement

                if max_count == 0:  # no match found in trigram
                    for replacement, _ in optional_replacements:
                        first_sum, second_sum = 0, 0
                        if ind != 0:
                            first_part_to_search = ' '.join(tokens[ind - 1: ind + 1]).replace(word_to_change, replacement)
                            first_sum = n_gram_model.n_tokens_counters[1].get(first_part_to_search, 0)
                        if ind != tokens_len - 1:
                            sec_part_to_search = ' '.join(tokens[ind: ind + 2]).replace(word_to_change, replacement)
                            second_sum = n_gram_model.n_tokens_counters[1].get(sec_part_to_search, 0)
                        count = first_sum + second_sum
                        if count > max_count:
                            max_count = count
                            best_match = replacement

            # Create new line
            pattern = re.compile(word_to_change, re.IGNORECASE)
            new_line = pattern.sub(best_match, line)
            output_str += new_line + '\n'

    print('Grammy Task - Done.')
    return output_str


def tokenize_sentence(sentence):
    # Adjust token and tokenize it
    sentence = sentence.replace('…', '...').replace('’', "'").replace('“', '"')
    for sign in r"""!"#$%&()*+,-./:;<=>?@[\]^_`{|}~""":
        sentence = sentence.replace(sign, ' ' + sign + ' ')
    sentence = sentence.replace('.  .  .', '...').replace('\'', ' \'').replace('n \'t', ' n\'t')
    tokens = list(filter(lambda token: token != '', sentence.split(' ')))
    return tokens


def tweets(tweets_file, model):
    print('Tweets Task - In Progress...')
    tweets_list = []
    # Parse tweets file
    with open(tweets_file) as file:
        current_category = None
        current_ind = 1
        for line in file:
            if '==' in line:
                current_category = line.replace('=', '').replace('\n', '').strip()
                current_ind = 1
            elif line.replace('\n', '') != '':
                new_tweet = Tweet(line.replace('\n', ''), current_category, current_ind)
                tweets_list.append(new_tweet)
                current_ind += 1

    words_scores = calculate_distance_based_score(tweets_list)
    weight_functions = {
        'Arithmetic': lambda _: 1,
        'Random': lambda _: randrange(10),
        'Custom - Distance Based': lambda token: words_scores.get(token, 0)
    }
    for weight_function_name, weight_function in weight_functions.items():
        # reduce dimension
        # todo on all data or per cat
        pca = PCA(n_components=2)
        all_vectors = [tw.get_new_vector(weight_function, model) for tw in tweets_list]
        pca.fit(all_vectors)
        adjusted = pca.transform(all_vectors)

        # plot results
        plt.title(f'{weight_function_name} - Shahar Stahi')
        for i, (x_point, y_point) in enumerate(adjusted):
            colors = {'Covid': 'blue', 'Olympics': 'red', 'Pets': 'green'}
            plt.scatter(x_point, y_point, color=colors[tweets_list[i].category], s=8)
            plt.text(x_point + .03, y_point + .03, f'{tweets_list[i].category}-{tweets_list[i].index}', fontsize=7)
        plt.show()
    print('Tweets Task - Done.')


def calculate_distance_based_score(tweets_list):
    covid_words = []
    olympics_words = []
    pets_words = []
    for tw in tweets_list:
        if tw.category == 'Covid':
            covid_words.extend(tw.tokens)
        elif tw.category == 'Olympics':
            olympics_words.extend(tw.tokens)
        else:
            pets_words.extend(tw.tokens)

    covid_words = np.array(covid_words)
    olympics_words = np.array(olympics_words)
    pets_words = np.array(pets_words)

    # get unique words and their count for each class
    covid_unique, covid_counts = np.unique(covid_words, return_counts=True)
    olympics_unique, olympics_counts = np.unique(olympics_words, return_counts=True)
    pets_unique, pets_counts = np.unique(pets_words, return_counts=True)

    covid_dictionary = dict(zip(covid_unique, covid_counts))
    olympics_dictionary = dict(zip(olympics_unique, olympics_counts))
    pets_dictionary = dict(zip(pets_unique, pets_counts))

    # give each word a score
    scores_dict = {}
    for word in np.unique(np.concatenate([covid_words, olympics_words, pets_words])):
        scores_dict[word] = abs(covid_dictionary.get(word, 0) - olympics_dictionary.get(word, 0)) + abs(
            covid_dictionary.get(word, 0) - pets_dictionary.get(word, 0)) + abs(
            olympics_dictionary.get(word, 0) - pets_dictionary.get(word, 0))

    return scores_dict


def print_output(output_file, task_a_str, task_b_str):
    output_str = task_a_str + task_b_str
    print(f'Writing output to {output_file}')
    with open(output_file, 'w', encoding='utf8') as output_file:
        output_file.write(output_str)


if __name__ == "__main__":
    main()
