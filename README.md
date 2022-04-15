# NATURAL LANGUAGE PROCESSING: ENGLISH TO FRENCH MACHINE TRANSLATION WITH TRANSFORMER

### Data Set Description
The data was gotten from [here](https://www.manythings.org/anki/fra-eng.zip), from the Tatoeba Project.
The text file contains one translation example per line: An English sentence, phrase, or word followed by a tab space, the corresponding translation in French, and finally an attribution message.  
There are a total of 192341 lines in the data set.

** Terms of Use **

See the terms of use.  
These files have been released under the same license as the
source.

http://tatoeba.org/eng/terms_of_use  
http://creativecommons.org/licenses/by/2.0  

Attribution: www.manythings.org/anki and tatoeba.org


### Data Preprocessing and Preparation
To preprocess the data for the Transformer model, standard text data preprocessing steps for Natural Language Processing (NLP) will be employed.  
First, every character in the text file will be normalized by converting them to unicode using the Unicode Normalization Form Compatibility Composition (NFKC) format.  
Then, the English and French sentences in each line in the text file will be separated out.  
In the text file, English words with more than one French translation have each pair of English to French translation in their own lines. These instances would be considered as duplicates. In the case of duplicates, only one of the English to French translations will be retained, the remaining will be dropped from the data set. This is done to keep things simple.
"[start]" and "[end]" tokens will be added to the start and end of each French translation. These will be used by the model to determine the beginning and end of each French sentence.

The next series of preparation steps will be to vectorize the text data. This will involve _standardizing_ the text to make it easier to process by converting all English and French characters to lowercase, and removing punctuations.  
Next is _tokenization_ where each sentence will split into units (tokens). The sentences will be split on whitespaces. In this case the tokens will be individual words.  
What follows is vocabulary indexing. This is when each token is encoded into a numerical representation. A vocabulary for all the tokens present in the training data will be built, and then a unique integer will be assigned to each of these tokens present in the vocabulary.
By convention, the first two entries in the vocabulary are the mask token, at index 0, and the Out of Vocabulary (OOV) token, at index 1. The OOV token will be used to represent new words that the model might encounter but are not present in the built vocabulary.
The standardizing, tokenization and vocabulary indexing steps will be handled by the Keras layer `TextVectorization`.


## DESCRIBING THE TRANSFORMER ARCHITECTURE
Transformer is a deep learning sequence model architecture introduced in the paper “Attention is all you need” by Vaswani et al, and soon after its introduction the transformer became widely adopted for most natural language processing (NLP) tasks, such as text classification, sentiment analysis, machine translation, etc.
The paper proposed that “a simple mechanism called ‘neural attention’ could be used to build powerful sequence models that did not feature any recurrent layers or convolution layers. (Chollet, 2021, p. 336).
Neural attention is the idea that a model should “pay more attention” to the input features that more significant / relevant and less attention to the others, because not all of the input information is equally important to the task at hand. 

Similar concepts to neural attention are Max Pooling in convolutional neural networks, and TF-IDF (Term Frequency - Inverse Document Frequency) normalization. The form of attention in the former case can be used to highlight certain important features in images while in the latter case, the form of attention employed there is used to give importance scores to tokens based on how much information they are expected to carry. While there are many different forms of attention, it is noteworthy that they all start by calculating and assigning importance scores for the features; more significant features are assigned higher scores, while less significant features are assigned lower scores.


### Word Embeddings
A Word embedding is simply a number or vector representation of a word. In the process of creating word embeddings, “the geometric relationship between two word vectors should reflect the semantic relationship between these words” (Chollet, 2021, p. 329).  This simply means that words that have similar meanings should also have similar word vectors.  For example football and soccer could refer to the same sport and hence be used interchangeably, therefore they should have vectors whose geometric distance between them are close.
Word embeddings are dense representations. Word embeddings are commonly 265-dimensional, 512-dimensional or 1,024-dimensional vectors.
Word embeddings are also structured representations. The structure of the representations is learned from the data, and thus “similar words get embedded in close locations, and further, specific directions in the embedding space are meaningful”. (Chollet, 2021, p. 330).

According to the paper, “an attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.” (Vaswani et al., 2017).

Beyond the stated applications of attention mechanism, neural attention can be used to make features context-aware. “The purpose of self-attention is to modulate the representation of a token by using the representations of related tokens in the sequence. This produces context aware token representations.” (Chollet, 2021, p. 338).  
Attention makes features context aware by computing the vector representations of each word based on its compatibility with the other words surrounding it.
This bit is very important for the subject of NLP, because inherently in natural languages (human languages), the meaning of words is context-specific and not universally fixed. 
Clarifying this with an example, consider the following sentences: “The Tennis player dropped the ball.” Considering one word in this sentence: ball. What kind of ball is being referred to? A basketball? A golf ball? Or probably a dance Ball? Self-attention algorithmically figures out that the “ball” in question is in fact a Tennis ball. It does this by  first computing attention scores between “ball” and every other word in the sentence. Then these attention scores are then used to weight the sum of all word vectors in the sentence. The words that are closely related to “ball” have greater weights, and thus contribute more to the sum of word vectors. The resulting vector is a context-aware representation for the word “ball” because it incorporates the surrounding context into its representation.
This process will be repeated for every word in the sentence.

In the transformer model, word vector representations are smart representations, i.e., they change according to the context in which the word is used to convey a piece of information.


### Multi-head attention
Multi-head attention is a tweak of a single attention function. It involves making linear projections of the queries, keys and values into separate subspaces resulting in three separate vectors, and on each of these separate projections, the attention function is independently performed in parallel, and the resulting vectors are then concatenated, and once again projected, resulting in a single sequence.
“Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions” (Vaswani et al., 2017).


### The Transformer Encoder
The Transformer encoder is one of the main parts of the Transformer architecture. It is used to process the source sequence, and its output is fed into Transformer decoder which uses the processed source sequence to generate a translated version of the sequence. However, the Transformer encoder as a standalone can be used for text classification.
The transformer encoder comprises of a Multi-head attention layer, a normalization layer (this helps the gradient flow better during backpropagation), a dense projection, residual connections and a second normalization layer.


### Positional Encoding
The Transformer encoder on its own is word-order agnostic i.e., it learns without considering the sequential ordering of the words in a sentence. This is contrary to the assertion that a Transformer is a sequential model and so to resolve this, word order information of each word is manually injected into its vector representation by a process called Positional Encoding.
The result of this is the input word embeddings will have two components: the usual word vector, which represents the word independently of any specific context, and a position vector, which represents the position of the word in the current sentence.
In this case, positional encoding will be implemented by learning the position-embedding vectors in the same way the word embeddings are learnt.
Positional encoding will be implemented in the model by a Positional Embedding Layer. This will be used just like a regular Embedding Layer.


### The Transformer Decoder
The Transformer decoder is the second main part of the Transformer model. The decoder processes the target sequence as well as the output of the Transformer encoder to generate a translation of the source sequence.
The architecture of the decoder is similar in structure to the Transformer Encoder, what distinguishes it from the encoder is that it has an extra attention block that takes as its query parameter the output of the first attention block (the output of self-attention on the target sequence), while its key and value parameters are the encoder output (processed source sequence) respectively.
Together the Transformer encoder and Transformer decoder make up the Transformer architecture.

In this project, the end-to-end Transformer model will be implemented with Tensorflow using the Functional API


### Examples of the Translations of the Model
English: Tom is the one who knows what needs to be done.
French Translation: [start] tom est qui sache ce sait ce qui doit être fait [end]

English: He was too drunk to drive home.
French Translation: [start] il était trop tombé à [UNK] pour rentrer à la maison [end]

English: She always buys expensive clothes.
French Translation: [start] elle compte toujours que de vêtements propre [end]

English: I could hardly believe my ears when I heard the news.
French Translation: [start] je pouvais à peine me comprendre qui les oreilles quand je [UNK] [end]

### References
Chollet, F. (2021). Deep Learning in Python (2nd edition). New York, Shelter Island: Manning Publications.

Ashish Vaswani et al., “Attention is all you need” (2017), https://arxiv.org/abs/1706.03762.