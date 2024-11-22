# import sys
# import os
# from nltk.parse.corenlp import CoreNLPParser
# from nltk.tree import ParentedTree, Tree  
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# import nltk
# import numpy as np
# import imageio
# from moviepy.editor import VideoFileClip, concatenate_videoclips

# def convert_to_isl(text):
#     nltk.download('punkt_tab')
#     nltk.download('averaged_perceptron_tagger_eng')

#     java_path = "C:\\Program Files\\Java\\jdk-21\\bin\\java.exe"
#     os.environ['JAVA_HOME'] = java_path

#     inputString = text

#     parser = CoreNLPParser(url='http://localhost:9000')

#     englishtree = list(parser.raw_parse(inputString))

#     parsetree = englishtree[0]

#     tree_dict = {}

#     parenttree = ParentedTree.convert(parsetree)

#     isltree = Tree('ROOT', [])
#     i = 0

#     for sub in parenttree.subtrees():
#         tree_dict[sub.treeposition()] = 0

#     for sub in parenttree.subtrees():
#         if sub.label() == "NP" and tree_dict[sub.treeposition()] == 0 and tree_dict[sub.parent().treeposition()] == 0:
#             tree_dict[sub.treeposition()] = 1
#             isltree.insert(i, sub)
#             i += 1
#         elif sub.label() in ["VP", "PRP"]:
#             for sub2 in sub.subtrees():
#                 if (sub2.label() in ["NP", "PRP"] and 
#                     tree_dict[sub2.treeposition()] == 0 and 
#                     tree_dict[sub2.parent().treeposition()] == 0):
#                     tree_dict[sub2.treeposition()] = 1
#                     isltree.insert(i, sub2)
#                     i += 1

#     for sub in parenttree.subtrees():
#         for sub2 in sub.subtrees():
#             if (len(sub2.leaves()) == 1 and 
#                 tree_dict[sub2.treeposition()] == 0 and 
#                 tree_dict[sub2.parent().treeposition()] == 0):
#                 tree_dict[sub2.treeposition()] = 1
#                 isltree.insert(i, sub2)
#                 i += 1

#     parsed_sent = isltree.leaves()
#     stop_words = set(stopwords.words("english"))
#     lemmatizer = WordNetLemmatizer()
#     lemmatized_words = []

#     for w in parsed_sent:
#         lemmatized_words.append(lemmatizer.lemmatize(w))

#     islsentence = ""
#     for w in lemmatized_words:
#         if w not in stop_words:
#             islsentence += w + " "

#     print(lemmatized_words)
#     print(islsentence)
#     try:
#         os.remove("my_concatenation.mp4")
#     except FileNotFoundError:
#         pass

#     name = islsentence.strip()
#     text = nltk.word_tokenize(name)
#     result = nltk.pos_tag(text)

#     for each in result:
#         print(each)

#     dict = {}
#     dict["NN"] = "noun"
#     arg_array = []

#     # Append video clips based on the tokenized text
#     for text in result:
#         video_folder = "E:/1_BTech/Major_Project/Developement-Code/FINAL/Text_to_ISL/islVideos2/"
#         found = False
#         # print(os.listdir(video_folder))
#         for video_file in os.listdir(video_folder):
#             print(video_file)
#             if text[0].lower() in video_file.lower():
#                 video_file_path = os.path.join(video_folder, video_file)
#                 arg_array.append(VideoFileClip(video_file_path))
#                 print(f"Video found: {video_file_path}")
#                 found = True
#                 break  

#         if not found:
#             print(f"Video not found for: {text[0]}")
      

#     if arg_array:
#         final_clip = concatenate_videoclips(arg_array)
#         final_clip.write_videofile("my_concatenation.mp4")
#     else:
#         print("No video clips found to concatenate.")


import sys
import os
from nltk.parse.corenlp import CoreNLPParser
from nltk.tree import ParentedTree, Tree  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import numpy as np
import imageio
from moviepy.editor import VideoFileClip, concatenate_videoclips

def convert_to_isl(text):
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger_eng')



    # Set up the Java path for Stanford CoreNLP
    java_path = "C:\\Program Files\\Java\\jdk-21\\bin\\java.exe"
    os.environ['JAVA_HOME'] = java_path

    # Initialize inputString
    inputString = text

    # # Combine command line arguments into a single input string
    # for each in range(1, len(sys.argv)):
    #     inputString += sys.argv[each] + " "

    # Start the CoreNLPParser
    parser = CoreNLPParser(url='http://localhost:9000')

    # Parse the input string
    englishtree = list(parser.raw_parse(inputString))

    # Get the first parse tree
    parsetree = englishtree[0]

    # Initialize a dictionary to track subtree positions
    tree_dict = {}

    # Convert to a parented tree for easier manipulation
    parenttree = ParentedTree.convert(parsetree)

    # Initialize isltree
    isltree = Tree('ROOT', [])
    i = 0

    # Populate tree_dict with initial values
    for sub in parenttree.subtrees():
        tree_dict[sub.treeposition()] = 0

    # Process the parse tree for NP and VP
    for sub in parenttree.subtrees():
        if sub.label() == "NP" and tree_dict[sub.treeposition()] == 0 and tree_dict[sub.parent().treeposition()] == 0:
            tree_dict[sub.treeposition()] = 1
            isltree.insert(i, sub)
            i += 1
        elif sub.label() in ["VP", "PRP"]:
            for sub2 in sub.subtrees():
                if (sub2.label() in ["NP", "PRP"] and 
                    tree_dict[sub2.treeposition()] == 0 and 
                    tree_dict[sub2.parent().treeposition()] == 0):
                    tree_dict[sub2.treeposition()] = 1
                    isltree.insert(i, sub2)
                    i += 1

    # Insert leaves into isltree
    for sub in parenttree.subtrees():
        for sub2 in sub.subtrees():
            if (len(sub2.leaves()) == 1 and 
                tree_dict[sub2.treeposition()] == 0 and 
                tree_dict[sub2.parent().treeposition()] == 0):
                tree_dict[sub2.treeposition()] = 1
                isltree.insert(i, sub2)
                i += 1

    # Extract parsed sentence and process it
    parsed_sent = isltree.leaves()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = []

    # Lemmatization and stopword removal
    for w in parsed_sent:
        lemmatized_words.append(lemmatizer.lemmatize(w))

    islsentence = ""
    for w in lemmatized_words:
        if w not in stop_words:
            islsentence += w + " "

    # Output the lemmatized words and final sentence
    print(lemmatized_words)
    print(islsentence)
    try:
        os.remove("my_concatenation.mp4")
    except FileNotFoundError:
        pass

    # Prepare input for video generation
    name = islsentence.strip()  # Use the processed ISL sentence
    text = nltk.word_tokenize(name)
    result = nltk.pos_tag(text)

    # Print part-of-speech tagging
    for each in result:
        print(each)

    # Create a dictionary for parts of speech
    dict = {}
    dict["NN"] = "noun"
    arg_array = []

    # Append video clips based on the tokenized text
    for text in result:
        video_folder = "E:/1_BTech/Major_Project/Developement-Code/FINAL/Text_to_ISL/islVideos2/"
        found = False
        for video_file in os.listdir(video_folder):
            print(video_file)
            # Check if any phrase in the video name matches with tokenized text (case-insensitive)
            if text[0].lower() in video_file.lower():
                video_file_path = os.path.join(video_folder, video_file)
                arg_array.append(VideoFileClip(video_file_path))
                print(f"Video found: {video_file_path}")
                found = True
                break  

        if not found:
            print(f"Video not found for: {text[0]}")

    if arg_array:
        final_clip = concatenate_videoclips(arg_array)
        final_clip.write_videofile("my_concatenation.mp4")
    else:
        print("No video clips found to concatenate.")
