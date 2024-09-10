import sys
import text_extracter

text = ""

if len(sys.argv) != 2:
    print("Not Enough Arguments Passed!")
    sys.exit()

text = text_extracter.text_getter(sys.argv[1])
if text == "":
    print("File Empty!")
    sys.exit()    

from mcq_generator import get_mca_questions
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    final_questions = get_mca_questions(text)
    for q in final_questions:
        print(q)
if __name__ == "__main__":
    main()