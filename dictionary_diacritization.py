#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
from sklearn.neural_network import MLPClassifier
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

class Dictionary:
    def __init__(self,
                 name="fiction-dictionary.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dictionary to `variants`
        self.variants = {}
        with open(name, "r", encoding="utf-8-sig") as dictionary_file:
            for line in dictionary_file:
                nodia_word, *variants = line.rstrip("\n").split()
                self.variants[nodia_word] = variants

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"),
                                       filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

def accuracy(gold: str, system: str) -> float:
    assert isinstance(gold, str) and isinstance(system, str), "The gold and system outputs must be strings"

    gold, system = gold.split(), system.split()
    assert len(gold) == len(system), "The gold and system outputs must have same number of words: {} vs {}.".format(len(gold), len(system))

    words, correct = 0, 0
    for gold_token, system_token in zip(gold, system):
        words += 1
        correct += gold_token == system_token

    return correct / words

def dictionarized(input, dict):
    splitted = input.split()
    dictedOutput = ""
    for word in splitted:
        noDiaWord = makeNoDia(word)
        if noDiaWord in dict.variants.keys():
            variants = dict.variants[noDiaWord]
            dictedOutput += chooseClosestVariant(variants,word) + " "
        else:
            dictedOutput +=  word + " "
    return dictedOutput[:-1]

def chooseClosestVariant(variants,word):
    distances = np.zeros(len(variants))
    for i in range(len(variants)):
        distances[i] = getDistance(variants[i],word)

    closest = np.where(distances == distances.min())

    return variants[closest[0][0]]

def getDistance(variant,word):
    differentLetters=0
    for i in range(len(word)):
        if(word[i]!=variant[i]):
            differentLetters+=1
    return differentLetters

def makeNoDia(word):
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"
    newWord=""
    for i in range(len(word)):
        upper = False
        lowered = word[i].lower()
        if(word[i].isupper()):
            upper = True
        else:
            upper = False

        if lowered in LETTERS_DIA:
            index = LETTERS_DIA.index(word[i])
            if(upper):
                newWord+=(LETTERS_NODIA[index].upper())
            else:
                newWord+=(LETTERS_NODIA[index])
        else:
            if(upper):
                newWord+=(word[i].upper())
            else:
                newWord+=(word[i])
    return newWord

def doOneHot(window):
    pocetPismenek = 122
    out = np.zeros(len(window) * pocetPismenek)
    for i in range(len(window)):
        out[i * pocetPismenek + ord(window[i]) - 1] = 1
    return out


def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        oneHotted = []
        targetMatrix = []
        windowSize = 5
        for i in range(windowSize, len(train.data) - windowSize):
            if(train.data[i] in train.LETTERS_NODIA):
                OneWindow = doOneHot(train.data[i - windowSize:i + windowSize])
                oneHotted.append(OneWindow)
                targetMatrix.append(train.target[i])
        model = MLPClassifier(hidden_layer_sizes=(300, 150), batch_size='auto', learning_rate='constant', max_iter=10, verbose=True).fit(oneHotted, targetMatrix)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        windowSize = 5
        output=""
        for i in range(windowSize,len(test.data)-windowSize):
            if(test.data[i] in test.LETTERS_NODIA):
                OneWindow = doOneHot(test.data[i - windowSize:i + windowSize])
                res =model.predict([OneWindow]) 
                output+= res[0]
            else:
                output += test.data[i]
    
        print(accuracy(test.target,test.data[0:windowSize] + output + test.data[-windowSize:] ))
        FinalOutput=test.data[0:windowSize] + output + test.data[-windowSize:]
        
        initDictionary = Dictionary()

        return dictionarized(FinalOutput,initDictionary)
        #return dictionarized("Praskání Měla Melá jg jablko",initDictionary)