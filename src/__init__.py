from typing import List, Union, Tuple, Optional


class G:
    """Manages global constants and util functions."""
    
    from nltk.corpus import wordnet, stopwords
    from nltk.stem import WordNetLemmatizer

    # Constants
    wordnet    = wordnet
    lemmatizer = WordNetLemmatizer()
    stopwords  = set(stopwords.words("english")) \
                    | {"be", "am", "is", "are", "was", "were", "been", "first", "last"}
    
    @classmethod
    def univ_tag_to_wn_tag(cls, tag: str) -> Optional[str]:
        """Convert a universal POS tag to a wordnet POS tag"""
        
        mapping = {"VERB": cls.wordnet.VERB, 
                   "NOUN": cls.wordnet.NOUN,
                   "ADJ":  cls.wordnet.ADJ,
                   "ADV":  cls.wordnet.ADV}
        return mapping.get(tag)
    
    @staticmethod
    def get_univ_tags(tokens: List[str]) -> List[Tuple[str, str]]:
        """Get the universal POS tags for a list of tokens"""
        
        from nltk import pos_tag
        
        return pos_tag(tokens, tagset="universal")
    
    @staticmethod
    def sent_tokenize(text: str) -> List[str]:
        """Tokenize a text into sentences using nltk's sentence tokenizer"""
        
        from nltk.tokenize import sent_tokenize
        
        return sent_tokenize(text)

    @staticmethod
    def word_tokenize(sentence: str) -> List[str]:
        """Tokenize a sentence using nltk's word tokenizer"""
        
        from nltk.tokenize import word_tokenize
        
        return word_tokenize(sentence)
    
    @staticmethod
    def sent_tokenize(paragraph: str) -> List[str]:
        """Tokenize a paragraph into sentences using nltk's sentence tokenizer"""
        
        from nltk.tokenize import sent_tokenize
        
        return sent_tokenize(paragraph)
    
    @classmethod
    def lemmatize_single(cls, token: str) -> str:
        """Lemmatize the given token. Try all possible 
            lemmatization (n, v, a, r, s) until the input changes."""
            
        all_pos = ["n", "v", "a", "r", "s"]

        ret = token
        while (ret == token) and all_pos:
            ret = cls.lemmatizer.lemmatize(token, all_pos.pop(0))
        return ret

    @classmethod
    def is_stopword(cls, token: str) -> bool:
        """Check if a token is a stopword"""
        
        return token.lower() in cls.stopwords


def antonym(tokens:  Union[str, List[str]], 
            sub_all: bool = False, 
            sample:  bool = False,
            delim:   str = " ") -> Union[List[str], str]:
    """Given a list of tokens from a SINGLE sentence (auto tokenize if given a string), 
        find the all (of first) words that have antonyms according 
        to nltk wordnet synsets, and replace it with their antonyms.
        An indicator is also returned to indicate if any antonyms were found."""

    import random
    
    if isinstance(tokens, str):
        tokens = G.word_tokenize(tokens)

    ret = [t for t in tokens]
    for i, (token, tag) in enumerate(G.get_univ_tags(tokens)):
        if G.is_stopword(token):
            continue

        wn_tag   = G.univ_tag_to_wn_tag(tag)
        antonyms = [lemma.antonyms()[0].name() 
                        for synset in G.wordnet.synsets(token, pos=wn_tag)
                        for lemma in synset.lemmas()
                        if lemma.antonyms() != []]
        
        if len(antonyms) == 0:
            continue
        else:
            if sample:
                ret[i] = random.sample(antonyms, 1)[0]
            else:
                ret[i] = antonyms[0]

            if not sub_all:
                break

    # Check if we need to join the tokens back together
    if delim:
        ret = delim.join(ret)
    return ret


def add_not(utterance: Union[str, List[str]], 
            delim:     str = " ") -> Union[Tuple[List[str], bool], Tuple[str, bool]]:
    """Add a "not" to the first verb in the utterance."""
    
    from pattern import en
    from nltk import pos_tag
    
    if isinstance(utterance, str):
        utterance = G.word_tokenize(utterance)
    
    utterance = [t for t in utterance]
    tagged_utterance = pos_tag(utterance)
    for token, tag in tagged_utterance:            
        if tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
            try:
                i = utterance.index(token)
            except:
                continue

            if tag in ["VB", "VBG", "VBN"]:
                utterance.insert(i, "not")
            elif tag == "VBD":
                if token in ["was", "were"]:
                    utterance[i] += "n't"
                else:
                    present = en.conjugate(token, person=1, tense=en.PRESENT)
                    if present != token:
                        utterance[i: (i + 1)] = ["didn't", present]
                    else:
                        continue
            elif tag == "VBP":
                if token == "am":
                    utterance.insert(i + 1, "not")
                elif token == "are":
                    utterance[i] += "n't"
                else:
                    utterance[i: (i + 1)] = ["don't", token]
            elif tag == "VBZ":
                if token == "is":
                    utterance[i] += "n't"
                else:
                    present = en.conjugate(token, person=1, tense=en.PRESENT)
                    if present != token:
                        utterance[i: (i + 1)] = ["doesn't", present]
                    else:
                        continue

            # only need to replace one token for each utterance
            break
    
    # Check if we need to join the tokens back together    
    if delim:
        utterance = delim.join(utterance)
    return utterance
