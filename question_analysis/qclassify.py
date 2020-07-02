import os
import io
import json
import gzip
import argparse
import collections
from time import time
from nltk.tokenize import TweetTokenizer
import statistics
import copy
import stanza



class qclass():
    def __init__(self):
        super(qclass, self).__init__()

        self.tknzr = TweetTokenizer(preserve_case=False)
        self.nlp_s = stanza.Pipeline(lang='en', processors='tokenize,pos,ner', tokenize_no_ssplit=True)
        self.color_tokens = []
        self.shape_tokens = []
        self.size_tokens = []
        self.texture_tokens = []
        self.action_tokens = []
        self.spatial_tokens =[]
        self.number_tokens = []
        self.object_tokens = []
        self.super_tokens = []

        self.attribute_tokens=[]

        word_annotation = './data/mod_word_annotation'

        with open(word_annotation) as f:
            lines = f.readlines()
            for line in lines:
        #         print(line)
                word1,word2 = line.split('\t')
        #         print(word1,word2)
                word1 = word1.strip().lower()
                word2 = word2.strip().lower()
                if word2 == 'color':
                    self.color_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'shape':
                    self.shape_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'size':
                    self.size_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'texture':
                    self.texture_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'action':
                    self.action_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'spatial':
                    self.spatial_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'number':
                    self.number_tokens.append(word1)
                    self.attribute_tokens.append(word1)
                elif word2 == 'object':
                    self.object_tokens.append(word1)
                elif word2 == 'super-category':
                    self.super_tokens.append(word1)

    def que_classify_attribute(self, que):
		#To Classify based on Attribute, Object and Super-category 
        que = que.lower()
        tokens = self.tknzr.tokenize(que)
        cat = '<NA>'
        if cat == '<NA>':
            for tok in tokens:
                if tok in self.attribute_tokens:
                    cat = '<attribute>'
                    break
            if cat == '<NA>':
                for tok in tokens:
                    if tok in self.object_tokens:
                        cat = '<object>'
                        break
                if cat == '<NA>':
                    for tok in tokens:
                        if tok in self.super_tokens:
                            cat = '<super-category>'
                            break
        return  cat

    def que_classify_multi(self, que):
		# Question Classification 
        que = que.lower()
        tokens = self.tknzr.tokenize(que)
        cat = '<NA>'
        if cat == '<NA>':
            tmp_cat=[]
            for tok in tokens:
                if tok in self.color_tokens:
                    tmp_cat.append('<color>')
                    break
            for tok in tokens:
                if tok in self.shape_tokens:
                    tmp_cat.append('<shape>')
                    break
            for tok in tokens:
                if tok in self.action_tokens:
                    tmp_cat.append('<action>')
                    break
            for tok in tokens:
                if tok in self.size_tokens:
                    tmp_cat.append('<size>')
                    break
            for tok in tokens:
                if tok in self.texture_tokens:
                    tmp_cat.append('<texture>')
                    break
            for tok in tokens:
                if tok in self.action_tokens:
                    tmp_cat.append('<action>')
                    break
            for tok in tokens:
                if tok in self.spatial_tokens or tok in self.number_tokens:
                    tmp_cat.append('<spatial>')
                    break

            if tmp_cat:
                cat = tmp_cat
            if cat == '<NA>':
                for tok in tokens:
                    if tok in self.object_tokens:
                        cat = '<object>'
                        break
                if cat == '<NA>':
                    for tok in tokens:
                        if tok in self.super_tokens:
                            cat = '<super-category>'
                            break


        return cat

    def cascade_classifier(self, batch):
        """
        Perform a classification of a question in a cascade fashion:
        if que is a number question:
           return number_question
        else if que is a relational question:
           return relational_question
        else
           return absolute_question

        Parameters:
        -----------
          batch: a string with a batch of questions to classify
                 separated by \\n\\n.

        Returns:
        --------
          a list of the location question type for each
          question in batch.
        """
        verbs = {'has', 'have', 'facing'}
        batch = batch.lower()
        # Perform POS with Stanza
        doc = self.nlp_s(batch)
        # List of location question types to return
        classification = []
        for sentence in doc.sentences:
            flags = []
            i_rel = -1
            i_verb = -1
            # Search for indicators
            for i, word in enumerate(sentence.words):
                if word.pos =='ADP' or word.text in verbs:
                    flags.append('rel')
                    if word.pos == 'ADP':
                        i_rel = i
                    if word.text in verbs:
                        i_verb = i
                if word.pos =='NUM':
                    flags.append('cou')

            # Classify
            # TODO this can be improved 
            if 'cou' in flags or self._has_number(sentence):
                classification.append('number')
                continue
            elif 'rel' in flags and (self._is_rel_adp(sentence, i_rel) or self._is_rel_verb(sentence, i_verb)):
                classification.append('relational')
            else:
                # If no NUM nor ADP seen, then we
                # say it is an absolute question
                classification.append('absolute') 

        return classification


    def _has_number(self, sentence):
        """
        Confirm if a question is number.
        """
        confirmation = False
        for ent in sentence.entities:
            if ent.type == 'ORDINAL' or ent.type =='CARDINAL':
                confirmation = True
                break
        return confirmation


    def _is_rel_adp(self, sentence, i_rel):
        """
        Confirm if a question is relational by looking at an
        adposition.
        """
        confirmation = False
        if i_rel != -1:
            # We have seen an ADP previously
            for word in sentence.words[i_rel:]:
                obj_cond = (word.text in self.object_tokens) or (word.text in self.super_tokens)
                if obj_cond:
                    confirmation = True
                    break
        return confirmation


    def _is_rel_verb(self, sentence, i_verb):
        """
        Confirm if a question is relational by looking at
        its verb.
        """
        confirmation = False
        if i_verb != -1:
            # We have seen a has/have/facing verb previously
            for word in sentence.words[i_verb:]:
                obj_cond = (word.text in self.object_tokens) or (word.text in self.super_tokens)
                if obj_cond:
                    confirmation = True
                    break
        return confirmation


    def location_classifier(self,que):
        """
        Classify location questions.
        [DEPRECATED]
        """
        que = que.lower()
        doc = self.nlp_s(que)
        flag = False
        #flags = [False for _ in doc.sentences]
        for sentence in doc.sentences:
            for i, word in enumerate(sentence.words):
                if word.pos =='ADP':
                    break
 
            for word in sentence.words[i:]:
                obj_cond = (word.text in self.object_tokens) or (word.text in self.super_tokens)
                flag = (word.pos == 'NOUN') and obj_cond
                if flag:
                    break
            #flags[j] = flag
        return flag


    def number_classifier_s(self,que):
        """
        Classify number questions.
        [DEPRECATED]
        """
        que = que.lower()
        doc = self.nlp_s(que)
        flag = False
        #flags = [False for _ in doc.sentences]
        for sentence in doc.sentences:
            flag = False
            for word in sentence.words:
                if word.pos =='NUM':
                    flag = True

        return flag
