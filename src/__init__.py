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


def antonym(tokens:     Union[str, List[str]], 
            is_sub_all: bool = True, 
            is_sample:  bool = False,
            join_sent:  bool = True) -> Union[Tuple[List[str], bool], Tuple[str, bool]]:
    """Given a list of tokens from a SINGLE sentence (auto tokenize if given a string), 
        find the all (of first) words that have antonyms according 
        to nltk wordnet synsets, and replace it with their antonyms.
        An indicator is also returned to indicate if any antonyms were found."""

    import random
    
    if isinstance(tokens, str):
        tokens = G.word_tokenize(tokens)
    
    alt = False
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
            alt = True
            if is_sample:
                ret[i] = random.sample(antonyms, 1)[0]
            else:
                ret[i] = antonyms[0]

            if not is_sub_all:
                break

    if join_sent:
        ret = " ".join(ret)
    return ret, alt
