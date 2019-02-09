
---

 This project is my solution to hackathon's data-science qualification task.

 Hackathon name is avoided to prevent plagiarism.

### Dependecies
---

*   python3
*   python3-opencv
*   python3-numpy
*   python3-matplotlib
*   tensorflow


### How to run
---

```
git clone https://github.com/klokik/RomanLetterClassificationTest.git
cd RomanLetterClassificationTest
python3 -m notebook RomanNumeralsClassification.ipynb
```

### Expected accuracy
---

Depending on your luck of how data samples would be splitted on training/test datasets, you may expect an accuracy in the range of `0.98` to `1.0`.
Dataset contains some quite badly written digits, and is tiny, therefore a few samples are expected to be be missclassified.

### Dataset info
---

Hand-drawn 136 samples of each "digit" on paper with a ball pen. Was [scanned](img001.jpg) and further processed with GIMP [_processed_](Variant2fixed4783.png) to be used in the script.

---

In case you can't or don't want to install `python3-opencv` you may download processed dataset from [gdrive](https://drive.google.com/open?id=1b-mTHwA7LpKv8xmNp_62rSTJ-82_dHXk) and read instructions in the notebook.
